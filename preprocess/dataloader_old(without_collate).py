import torch
from torch.utils.data import Dataset

import pandas as pd
import numpy as np



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
            semi_preprocess_df= None, 
            transform=False):
        '''
            train_df : ['path', 'file_id', 'sequence_id', 'participant_id', 'phrase']

            semi_preprocess_df : pd.DataFrame
        
        '''
        self.config = config
        self.df = train_df
        self.char_to_num = char_to_num
        self.transform = transform
        if semi_preprocess_df is not None:
            self.semi_preprocess_df = semi_preprocess_df.set_index("sequence_id")
        else:
            self.semi_preprocess_df = semi_preprocess_df

        self.right_hand = []
        self.left_hand = []
        for i in range(21):
            self.right_hand += [f'x_right_hand_{i}', f'y_right_hand_{i}', f'z_right_hand_{i}']
            self.left_hand += [f'x_left_hand_{i}', f'y_left_hand_{i}', f'z_left_hand_{i}']

    def transform_char_to_num(self, y):
        return torch.tensor([self.char_to_num[i] for i in y]).reshape(-1)

    def preprocess_frames(self, df):
        '''
            df : partial (train_landmarks or supplemental_landmarks) which contains 
                 frames, face, hand, pose data

                 compress sequential na sets
                 and replace na with zero

        '''
        check_na = pd.DataFrame(df.isna().sum(axis=1))
        check_na[1] = check_na[0].shift(1)
        check_na['remove'] = (check_na[0] == check_na[1]) & (check_na[0] != 0)
        final_df = df.drop(check_na[check_na['remove']].index, axis=0).fillna(0)
        return final_df

    def check_which_hands(self, right_df, left_df):
        if len(right_df) >= len(left_df):
            return right_df, 'right'
        else:
            return left_df, 'left'

    def make_features(self, df):
        '''
        make distance features & vector   per  every frame
        '''

        # make distance
        

        # calculate vectors
        distance_ls = []
        result_ls = []
        
        for frame in df:
            if np.isnan(frame).sum() > 0:
                continue

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

        if len(df) > max_len:

            na_idx = df[df.sum(axis=1) == 0].index.tolist()
            notna_idx = df[df.sum(axis=1) != 0].index.tolist()

            rest_len = max_len - len(na_idx)
            picked_notna_idx = np.random.choice(notna_idx, rest_len)
            final_idx = na_idx + picked_notna_idx.tolist()
            return df.loc[final_idx].reset_index(drop=True)

        else:
            number_of_pad_required = max_len - len(df)
            shape_in_row = df.shape[-1]
            zero_df = pd.DataFrame(np.zeros([number_of_pad_required, shape_in_row]))
            zero_df.columns = df.columns

            return pd.concat([df.reset_index(drop=True), zero_df], ignore_index=True)



    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, item):
        row = self.df.iloc[item]

        sequence_id = row['sequence_id'] 
        video_fn = self.config.root_path + row['path']

        y = row['phrase']
        y = y + (self.config.max_target_length - len(y))*'P'

        y = self.transform_char_to_num(y)

        if self.semi_preprocess_df is not None:
            frame_df = self.semi_preprocess_df.loc[[sequence_id]]
            frame_df = frame_df.set_index("frame")
        else:
            frame_df = pd.read_parquet(video_fn).set_index("sequence_id").loc[sequence_id]
            frame_df = frame_df.set_index("frame")

        right_df = frame_df[self.right_hand]
        left_df = frame_df[self.left_hand]

        right_df = self.preprocess_frames(right_df)
        left_df = self.preprocess_frames(left_df)

        '''
        1.hands 별로 model을 만들어야 하는것인지 -> 기각, frame길이가 안맞음
            -> 이 방법을 선택하려면, preprocess_frames(compress) 과정이 없어야함.

        2.hands가 선택되면, flip을 해야하나..


        일단 그냥 진행하자.
        '''
        hand_df, which_hand = self.check_which_hands(right_df, left_df)
        hand_df = hand_df.fillna(0) # preprocess_frames에서 진행해서 안해도 되긴함.
        hand_df = self.padding_or_subsampling(hand_df)

        frame_len = len(hand_df)

        # reshape to [frame, node, feature들] 
        hand_df = torch.tensor(hand_df.values.reshape(frame_len, -1, 3))
        hand_df = self.make_features(hand_df)

        if self.transform:
            '''
                which hand인지 따라, transform을 다르게 해야겟음.
            '''
            None ################ fill

        return {
            'y':y,
            'hand_df': hand_df
        }


if __name__ == "__main__":

    from dataclasses import dataclass, fields, asdict


    @dataclass
    class Config:

        version : str  # graph, cnn_like, transformer_like
        train_fn : str #  = './data/train.csv'
        root_path : str
        char2pred_fn : str

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
        train_fn = './data/train.csv',
        root_path = './data/',
        char2pred_fn = './data/character_to_prediction_index.json',

        valid_ratio= .2,
        gpu_id=0,
        batch_size=32,
        adam_epsilon=1e-8,
        use_radam= True,
        scheduler = False,
        warmup_ratio = .2,
        lr = 5e-5,

        max_frame_length = 200, ###########바꿔야지
        max_target_length = 45,
        num_target = 60, # including Pad token, and 'None' token
        n_epochs = 30,
        logging_fn = './log/training.log',
        model_fn = './ckpt/',
        load_model_fn = './ckpt/',

        random_seed = 42
        )
    
    
    import json
    from torch.utils.data import DataLoader
    import time

    train_possible = pd.read_csv("./data/train.csv")
    print(train_possible.head())
    partial_train_df = pd.read_parquet("./data/partial_train_landmark.parquet")

    with open ("./data/character_to_prediction_index.json", 'r') as f:
        char_to_num = json.load(f)
    def add_character(json_obj, 
                    tokens:list = ['P']):
        
        next_token = len(json_obj)
        for idx, token in enumerate(tokens):
            json_obj[token] = next_token
            next_token += 1

        return json_obj
    
    char_to_num = add_character(char_to_num)

    train_loader = DataLoader(
        ClassificationDataset(
            train_possible,
            char_to_num,
            config,
            semi_preprocess_df=partial_train_df
        ),
        batch_size = 32,
    )

    for i in range(2):
        start = time.time()
        print(next(iter(train_loader))['hand_df'].shape)
        end = time.time()
        print(end-start)