import torch
from torch.utils.data import Dataset

import pandas as pd
import numpy as np

class PadFill():

    def __init__(self, char_to_num, config, adj_matrix):
        '''
            tokenizer : hugging face tokeninzer
            max_length : limit sequence length of texts
            with_text : return with original text which means not passing through tokenizer

        '''
        self.max_frame_length = config.max_frame_length
        self.max_target_length = config.max_target_length
        self.config = config
        self.char_to_num = char_to_num
        self.adj_matrix = adj_matrix
        
    def padding_hands(self, frames):
        (_, node_len, feature_len) = frames[0].shape

        # max_frame = min(self.config.max_frame_length, max([i.shape[0] for i in frames]))
        max_frame = max([i.shape[0] for i in frames])
        if self.config.padding_max:
            max_frame = self.max_frame_length


        frames = torch.nested.nested_tensor(frames)
        frames = torch.nested.to_padded_tensor(frames, 0, (self.config.batch_size,
                                                           max_frame, 
                                                            node_len,
                                                            feature_len))
        return frames, max_frame

    def padding_target(self, targets):
        
        max_char = min(
            self.max_target_length,
            max([i.shape[0] for i in targets])
            )

        if self.config.padding_max:
            max_char = self.max_target_length

        if self.config.add_padding_token:
            targets = torch.nested.nested_tensor(targets)
            targets = torch.nested.to_padded_tensor(targets, 
                                                    self.char_to_num['P'], 
                                                    (self.config.batch_size,
                                                    max_char))
        else:
            targets = torch.nested.nested_tensor(targets)
            targets = torch.nested.to_padded_tensor(targets, 
                                                    len(self.char_to_num), 
                                                    (self.config.batch_size,
                                                    max_char))
        return targets, max_char
    

    def __call__(self, bs):
        # y, ytoken, hand_df

        frames = [b['hand_df'] for b in bs]    # bs = from ClassificationDataset
        targets = [b['y'] for b in bs]
        
        if self.config.padding == True:
            frames, max_frame = self.padding_hands(frames)
            targets, max_y = self.padding_target(targets)
            frame_length = torch.tensor(len(bs)*[max_frame]).long()
            y_length = torch.tensor(len(bs)*[max_y]).long()
        
        else:
            frame_length = torch.tensor([frame.shape[1] for frame in frames]).long()
            y_length = torch.tensor([target.shape[0] for target in targets]).long()


        return {
            'hand_df':frames,
            'adj_matrix':self.adj_matrix,
            'y':targets,
            'frame_length':frame_length,
            'y_length':y_length,
            }



class ClassificationDataset(Dataset):
    '''
    how to duplicate or how to get robustness by making more datasets
        1. flip right side to left  and  put left hands to right model vise versa
        2. tilt
    

    make more features
        1. calculate direction vector  : 21 combination 2 : but one is already fixed
        2. calculate distances vector : 21 combination 2 : 210 cases
    
    '''

    def __init__(
            self, 
            train_df, 
            char_to_num, 
            config, 
            right_hand,
            left_hand,
            semi_preprocess_df= None, 
            transform=False,
            drop = False):
        '''
            train_df : ['path', 'file_id', 'sequence_id', 'participant_id', 'phrase']

            semi_preprocess_df : frame,,  x,y,z 
        
        '''
        self.config = config
        self.df = train_df
        self.char_to_num = char_to_num
        self.transform = transform
        self.drop = drop
        if semi_preprocess_df is not None:
            self.semi_preprocess_df = semi_preprocess_df.set_index("sequence_id") if 'sequence_id' in semi_preprocess_df.columns else semi_preprocess_df
        else:
            pass
            # self.semi_preprocess_df = semi_preprocess_df
        self.semi_preprocess_df_right_hand = self.preprocess_frames(semi_preprocess_df[right_hand])
        self.semi_preprocess_df_left_hand = self.preprocess_frames(semi_preprocess_df[left_hand]) # sequence_id is set as index

        try:
            self.semi_preprocess_df_left_hand = self.semi_preprocess_df_left_hand.set_index("sequence_id")
            self.semi_preprocess_df_right_hand = self.semi_preprocess_df_right_hand.set_index("sequence_id")
        except:
            pass

        self.right_hand_sequence = self.semi_preprocess_df_right_hand.reset_index().sequence_id.unique()
        self.left_hand_sequence = self.semi_preprocess_df_left_hand.reset_index().sequence_id.unique()
        
        # path = self.df['path'][0]
        # frame_df = pd.read_parquet(path)
        # self.frame_df = frame_df.set_index("frame")

        self.right_hand = right_hand
        self.left_hand = left_hand

    def transform_char_to_num(self, y):
        return torch.tensor([self.char_to_num[i] for i in y]).reshape(-1)

    def preprocess_frames(self, df):
        '''
            df : partial (train_landmarks or supplemental_landmarks) which contains 
                 frames, face, hand, pose data

                 compress sequential na sets
                 and replace na with zero

        '''
        if self.drop:
            df = df.reset_index()
            df = df[df.notna()]
            if 'sequence_id' not in df.columns:
                df = df.set_index("sequence_id")
            return df
        else:
            df = df.reset_index()
            check_na = pd.DataFrame(df.isna().sum(axis=1))
            check_na[1] = check_na[0].shift(1)
            check_na['remove'] = (check_na[0] == check_na[1]) & (check_na[0] != 0)
            final_df = df.iloc[check_na[~check_na['remove']].index]
            # final_df = df.drop(check_na[check_na['remove']].index, axis=0).fillna(0)
            if 'sequence_id' not in final_df.columns:
                final_df = final_df.set_index("sequence_id")
            return final_df


    def check_which_hands(self, right_df, left_df):
        if right_df.isna().values.sum() >= left_df.isna().values.sum():
            return left_df, 'left'
        else:
            return right_df, 'right'
    
    def make_distance(self, df):
        return torch.cdist(df, df)
    
    def make_direction_vector(self, df):
        result_ls = []
        for frame in df:
            num_nodes = frame.shape[0]
            to_substract = torch.concat([frame.unsqueeze(0)]*num_nodes, dim=0)   # [node, node, 3]
            substracted = frame.unsqueeze(1) - to_substract                      # [node, 1, 3] - [node, node, 3]  : broadcasting -> [node, node, 3]
            substracted = substracted.reshape(num_nodes,-1).unsqueeze(0)         # [1, node, node*3]
            result_ls += [substracted]
        result = torch.concat(result_ls, dim=0)                                  # [frame, node, node*3]
        return result

    def make_features(self, df):

        '''
        deprecicated

        make distance features & vector   per  every frame
        '''

        # make distance
        

        # calculate vectors
        distance_ls = []
        result_ls = []
        
        for frame in df:
            # if np.isnan(frame).sum() > 0:
            #     continue

            distance_ls += [
                torch.cdist(frame, frame).unsqueeze(0)
            ]
            every_vectors_ls = []
            for node in frame:
                node_vector = []
                for other_node in frame:
                    node_vector += [(node-other_node).reshape(1,-1)]
                node_vector = torch.concat(node_vector, dim=-1)
                every_vectors_ls += [node_vector]

            every_vectors_ls = torch.concat(every_vectors_ls, dim=0)
            result_ls += [every_vectors_ls.unsqueeze(0)]

        distance = torch.concat(distance_ls, dim=0)
        result = torch.concat(result_ls, dim=0)
        df = torch.concat([df, distance, result], dim=-1)

        return df
            

    def padding_or_subsampling(self, df):
        max_len = self.config.max_frame_length

        if self.drop and len(df) > max_len:
            # notna_idx = df[df.isna().sum(axis=1) != 0].index.tolist()
            # rest_len = max_len - len(notna_idx)
            # if len(notna_idx) == 0:
                # return df.iloc[0].fillna(0).reset_index(drop=True)
            picked_notna_idx = np.random.choice(df.index.tolist(), max_len).tolist()
            output = df.loc[picked_notna_idx].reset_index(drop=True)
            return output

        elif self.drop:
            return df

        elif len(df) > max_len:
            na_idx = df[df.isna().sum(axis=1) == 0].index.tolist()
            notna_idx = df[df.isna().sum(axis=1) != 0].index.tolist()

            # rest_len = max_len - len(na_idx)
            rest_len = max(min(max_len - len(na_idx), max_len), max_len)
            picked_notna_idx = np.random.choice(notna_idx, rest_len)
            final_idx = na_idx + picked_notna_idx.tolist()
            output = df.loc[final_idx].reset_index(drop=True)
            return output

        else:
            # number_of_pad_required = max_len - len(df)
            # shape_in_row = df.shape[-1]
            # zero_df = pd.DataFrame(np.zeros([number_of_pad_required, shape_in_row]))
            # zero_df.columns = df.columns

            # return pd.concat([df.reset_index(drop=True), zero_df], ignore_index=True)
            return df


    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, item): # [frame, node, feature] <- frame, (node*feature)
        row = self.df.iloc[item]

        sequence_id = row['sequence_id'] 
        # video_fn = self.config.root_path + row['path']

        y = row['phrase']
        y_token = row['y_token']
        # y = y + (self.config.max_target_length - len(y))*'P'
        # y = self.transform_char_to_num(y)

        # if self.semi_preprocess_df is not None:
        #     frame_df = self.semi_preprocess_df.loc[[sequence_id]]
        #     frame_df = frame_df.set_index("frame")
        # else:
        #     frame_df = self.frame_df

        # right_df = frame_df[self.right_hand]
        # left_df = frame_df[self.left_hand]

        # right_df = self.preprocess_frames(right_df)
        # left_df = self.preprocess_frames(left_df)

        '''
        1.hands 별로 model을 만들어야 하는것인지 -> 기각, frame길이가 안맞음
            -> 이 방법을 선택하려면, preprocess_frames(compress) 과정이 없어야함.

        2.hands가 선택되면, flip을 해야하나..


        일단 그냥 진행하자.
        '''
        
        if (sequence_id in self.right_hand_sequence) and (sequence_id in self.left_hand_sequence):
            hand_df, which_hand = self.check_which_hands(
                self.semi_preprocess_df_right_hand.loc[sequence_id], 
                self.semi_preprocess_df_left_hand.loc[sequence_id]
            ) # [frame, node*3]

        elif sequence_id in self.right_hand_sequence:
            hand_df = self.semi_preprocess_df_right_hand
            which_hand = 'right'

        else:
            hand_df = self.semi_preprocess_df_left_hand
            which_hand = 'left'
        
        # hand_df = hand_df.fillna(0) # preprocess_frames에서 진행해서 안해도 되긴함.
        hand_df = hand_df.reset_index(drop=True)        # dropping, sequence_id 
        hand_df = self.padding_or_subsampling(hand_df)
        hand_df = hand_df.fillna(0)

        frame_len = len(hand_df) # frame

        # reshape to [frame, node, feature들] 
        
        if len(hand_df.shape)==1:
            hand_df = np.array(hand_df)
            hand_df = np.expand_dims(hand_df, axis=0)
            hand_df = pd.DataFrame(hand_df)
            frame_len = 1
        hand_df = torch.tensor(hand_df.values.reshape(frame_len, -1, 3))  # frame, node, 3
        distance = self.make_distance(hand_df)
        direction_vec = self.make_direction_vector(hand_df)
        hand_df = torch.concat([hand_df, distance, direction_vec], dim=-1)

        # hand_df = self.make_features(hand_df)

        if self.transform:
            '''
                which hand인지 따라, transform을 다르게 해야겟음.
            '''
            None ################ fill

        return {
            'y':torch.tensor(y_token).reshape(-1).long(),
            'y_label' : y,
            'hand_df': hand_df, # frame, node, feature(87)
            'sequence_id' : sequence_id
        }


if __name__ == "__main__":

    from dataclasses import dataclass, fields, asdict


    @dataclass
    class Config:

        version : str  # graph, cnn_like, transformer_like
        train_fn : list #  = './data/train.csv'
        root_path : str
        char2pred_fn : str
        semi_preprocess_df : list
        add_padding_token : bool
        ADJ_NORM : bool

        valid_ratio : float #= .2

        gpu_id : int
        batch_size : int
        adam_epsilon : float # = 1e-8
        use_radam : bool # = True
        scheduler : bool #False
        warmup_ratio : float #= .2
        lr : float #= 5e-5

        max_frame_length : int #= 30 ###########바꿔야지
        max_target_length : int
        num_target : int #= 59 # including Pad token
        n_epochs : int #= 30
        logging_fn :str #= './log/training.log'
        model_fn : str #= './ckpt/'
        load_model_fn : str # './ckpt/'
        random_seed : int

    config = Config(
        version = 'graph',
        train_fn = ['./data/train_possible.parquet', './data/supplemental_possible.parquet'],
        # semi_preprocess_df = None,
        semi_preprocess_df = ['./data/partial_train_landmark01.parquet', './data/partial_supplemental_landmark01.parquet'],
        root_path = './data/',
        char2pred_fn = './data/character_to_prediction_index.json',
        add_padding_token = False,
        ADJ_NORM = True,

        valid_ratio= .2,
        gpu_id=0,
        batch_size=2, ###########################
        adam_epsilon=1e-8,
        use_radam= True,
        scheduler = False,
        warmup_ratio = .2,
        lr = 5e-5,

        max_frame_length = 200, ###########바꿔야지
        max_target_length = 45,
        num_target = 59, # including Pad token, and 'None' token
        n_epochs = 30,
        logging_fn = './log/training.log',
        model_fn = './ckpt/',
        load_model_fn = './ckpt/',

        random_seed = 42
        )
    
    import json
    from torch.utils.data import DataLoader
    import time
    ADJ_MATRIX = torch.tensor([
        #0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0,  1,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  1,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1,  0,  0,  0,  0,  1,  0,  0,  1,  0,  0,  0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  1,  0,  1,  0,  0,  0,  0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  1,  0,  0,  0,  0,  1,  0,  0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0],
    ]).float()

    # adj matric handling
    if ADJ_MATRIX[0][0] != 1:
        ADJ_MATRIX += torch.eye(ADJ_MATRIX.shape[0])


    # random seed fix
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    # data handling
    train_df = []
    for train_fn in config.train_fn:
        train_df += [
            pd.read_parquet(train_fn)
        ]
    train_df = pd.concat(train_df)

    if config.semi_preprocess_df is not None:
        semi_df = []
        for semi_fn in config.semi_preprocess_df:
            print(pd.read_parquet(semi_fn).tail()) 
            semi_df += [
                pd.read_parquet(semi_fn)
            ]
        semi_df = pd.concat(semi_df)
    else:
        semi_df = config.semi_preprocess_df
        
    with open (config.char2pred_fn, 'r') as f:
        char_to_num = json.load(f)

    # add P token
    if config.add_padding_token == False:
        final_output = len(char_to_num)+1
    else:
        final_output = len(char_to_num) 


    train_loader = DataLoader(
        ClassificationDataset(
            train_df,
            char_to_num,
            config,
            semi_preprocess_df=semi_df
        ),
        batch_size = config.batch_size,
        shuffle=True,
        collate_fn=PadFill(char_to_num, config, ADJ_MATRIX)
    )

    print("-" * 100)
    for i in range(2):
        start = time.time()
        sample = next(iter(train_loader))
        print(f'hand_df : {sample["hand_df"].shape}')
        print(f'y : {sample["y"].shape}')
        print(sample['y'])
        end = time.time()
        print(end-start)


    # from transformers import BertModel, BertTokenizerFast
    # tokenizer = BertTokenizerFast.from_pretrained('kykim/bert-kor-base')

