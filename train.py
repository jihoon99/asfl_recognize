# train.py 
'''
    - data load
        - preprocess : dataloader
    - model : architechture
    - trainer : epoch
'''

import os
import argparse
import random
import pandas as pd
import numpy as np
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch_optimizer as custom_optim

from sklearn.model_selection import train_test_split

from dataclasses import dataclass, fields, asdict
from datetime import date

from pprint import pprint

from preprocess.dataloader import ClassificationDataset, PadFill
from model.GCN import GCNNet
from trainer import training, validating

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


@dataclass
class Config:

    version : str  # graph, cnn_like, transformer_like
    train_fn : list #  = './data/train.csv'
    frame_fn : list
    root_path : str
    char2pred_fn : str
    semi_preprocess_df : list
    add_padding_token : bool
    ADJ_NORM : bool
    padding : bool
    # num_df_parts : int
    drop_na : bool


    valid_ratio : float #= .2

    gpu_id : int
    batch_size : int
    adam_epsilon : float # = 1e-8
    use_radam : bool # = True
    scheduler : bool #False
    warmup_ratio : float #= .2
    lr : float #= 5e-5
    max_grad_norm : float
    num_layer : bool


    max_frame_length : int #= 30 ###########바꿔야지
    max_target_length : int
    num_target : int #= 59 # including Pad token
    n_epochs : int #= 30
    logging_fn :str #= './log/training.log'
    model_fn : str #= './ckpt/'
    load_model_fn : str # './ckpt/'
    random_seed : int


def preprocess_frames(df):
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


def add_character(json_obj, 
                  tokens:list = ['P']):
    
    next_token = len(json_obj)
    for idx, token in enumerate(tokens):
        json_obj[token] = next_token
        next_token += 1

    return json_obj

def train_valid_split(df, config):
    length_df = len(df)
    train_size = len(df)*config.valid_ratio


def get_loaders1(
        train_df,
        char_to_num : json,
        config,
        semi_preprocess_df = None,
        transform = None,
        adjacency_matrix=ADJ_MATRIX):
    import datetime


    print("-"*100)
    print("Loading DataLoader...")

    def tmp(x):
        return [char_to_num[i] for i in x]
    train_df['TARGET'] = train_df['phrase'].apply(tmp)

    num_to_char = {j:i for i,j in char_to_num.items()}

    train, valid = train_test_split(
        train_df, 
        test_size=config.valid_ratio, 
        random_state=config.random_seed)


    # split train, valid
    train=train_df.sample(frac=0.8,random_state=42)
    valid=train_df.drop(train.index)

    paths = train_df['path'].unique().tolist()

    train_loaders = []
    valid_loaders = []

    paths = train.path.unique().tolist()

    print("COOKING DATALOADER...")
    for idx, path in enumerate(paths):
        start = datetime.datetime.now()
        print(path, idx, end='\r')
        print(path)
        partial_train_df = train_df.set_index('path').loc[path].reset_index()
        print(partial_train_df.path.unique())
        partial_valid_df = train_df.set_index("path").loc[path].reset_index()
        print(partial_valid_df.path.unique())
        if len(partial_train_df) != 0:
            train_loaders += [DataLoader(
                ClassificationDataset(
                    partial_train_df,
                    char_to_num,
                    config,
                    semi_preprocess_df=pd.read_parquet(
                        config.root_path + partial_train_df['path'].iloc[0])[RIGHT_HAND + LEFT_HAND],
                    right_hand = RIGHT_HAND,
                    left_hand = LEFT_HAND,
                ),
                batch_size = config.batch_size,
                shuffle=True,
                collate_fn=PadFill(char_to_num, config, adjacency_matrix)
            )] 
 
        if len(partial_valid_df) != 0:
            valid_loaders += DataLoader(
                ClassificationDataset(
                    valid,
                    char_to_num,
                    config,
                    semi_preprocess_df=pd.read_parquet(
                        config.root_path + partial_valid_df['path'].iloc[0])[RIGHT_HAND + LEFT_HAND],
                    right_hand = RIGHT_HAND,
                    left_hand = LEFT_HAND,
                ),
                batch_size = config.batch_size,
                shuffle=True,
                collate_fn=PadFill(char_to_num, config, adjacency_matrix)
            )
        end = datetime.datetime.now()

        print(end-start)
    return train_loaders, valid_loaders, char_to_num, num_to_char






def get_loaders(
        train_df,
        char_to_num : json,
        config,
        frame_df = None,
        drop_na = True,
        transform = None,
        adjacency_matrix=ADJ_MATRIX):
    import datetime


    print("-"*100)
    print("Loading DataLoader...")

    def tmp(x):
        return [char_to_num[i] for i in x]

    num_to_char = {j:i for i,j in char_to_num.items()}

    train_loaders = []
    valid_loaders = []
    for t, f in zip(train_df, frame_df):
        _train = pd.read_parquet(t).reset_index()           # path, file_id, sequence_id, phrase
        _frame = pd.read_parquet(f)                         # index:sequenceid, x~,y~,z~
        _train['y_token'] = _train['phrase'].apply(tmp)

        # split train, valid
        _train_final=_train.sample(frac=1-config.valid_ratio, random_state=42)
        _valid_final=_train.drop(_train_final.index)


        print("COOKING DATALOADER...")
        start = datetime.datetime.now()
        train_loaders += [DataLoader(
            ClassificationDataset(
                _train_final,
                char_to_num,
                config,
                semi_preprocess_df=_frame[RIGHT_HAND + LEFT_HAND],
                right_hand = RIGHT_HAND,
                left_hand = LEFT_HAND,
                drop = drop_na
            ),
            batch_size = config.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=PadFill(char_to_num, config, adjacency_matrix)
        )] 
    
        valid_loaders += [DataLoader(
            ClassificationDataset(
                _valid_final,
                char_to_num,
                config,
                semi_preprocess_df=_frame[RIGHT_HAND + LEFT_HAND],
                right_hand = RIGHT_HAND,
                left_hand = LEFT_HAND,
                drop = drop_na
            ),
            batch_size = config.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=PadFill(char_to_num, config, adjacency_matrix)
        )]

        end = datetime.datetime.now()

        print(end-start)
    return train_loaders, valid_loaders, char_to_num, num_to_char





def normalizeAdjacency(W):
    """
    NormalizeAdjacency: Computes the degree-normalized adjacency matrix

    Input:

        W (np.array): adjacency matrix

    Output:

        A (np.array): degree-normalized adjacency matrix
    """
    # Check that the matrix is square
    assert W.shape[0] == W.shape[1]
    W = W + torch.eye(W.shape[0])
    # Compute the degree vector
    d = W.sum(axis = 1)
    # Invert the square root of the degree
    d = 1/np.sqrt(d)
    # And build the square root inverse degree matrix
    D = torch.diag(d)
    # Return the Normalized Adjacency
    return D @ W @ D 


def get_optimizer(model, config):
    '''
        set optimizer and return optimizer
    '''

    if config.use_radam:
        # if using RAdam
        optimizer = custom_optim.RAdam(model.parameters(), lr=config.lr)

    else:
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']    # no decay on bias and layerNorm
        optimizer_grouped_parameters = [
            # weight decay except no_decay
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=config.lr,
            eps=config.adam_epsilon
        )

    return optimizer

RIGHT_HAND = []
LEFT_HAND = []
for i in range(21):
    RIGHT_HAND += [f'x_right_hand_{i}', f'y_right_hand_{i}', f'z_right_hand_{i}']
    LEFT_HAND += [f'x_left_hand_{i}', f'y_left_hand_{i}', f'z_left_hand_{i}']



def transform_char_to_num(y):
    return [char_to_num[i] for i in y]

if __name__ == "__main__":
    config = Config(
        version            = 'graph',
        # train_fn           = ['./data/train_possible.parquet', './data/supplemental_possible.parquet'],
        # train_fn           = ['./data/partial_train_0.parquet', './data/partial_train_1.parquet', './data/partial_train_2.parquet'],
        # frame_fn           = ['./data/frame0.parquet','./data/frame1.parquet', './data/frame2.parquet'],
        train_fn           = ['./data/partial_train_0.parquet'],
        frame_fn           = ['./data/frame0.parquet'],
        
        semi_preprocess_df = None,
        # semi_preprocess_df = ['./data/partial_train_landmark01.parquet', './data/partial_supplemental_landmark01.parquet'],
        root_path          = './data/',
        char2pred_fn       = './data/character_to_prediction_index.json',
        add_padding_token  = False,
        ADJ_NORM           = True,
        padding            = True,
        # num_df_parts       = 10,
        drop_na            = True,

        valid_ratio        = .2,
        gpu_id             = 0,
        batch_size         = 32, ###########################
        adam_epsilon       = 1e-8,
        use_radam          = True,
        scheduler          = False,
        warmup_ratio       = .2,
        lr                 = 5e-5,
        max_grad_norm      = 5.,
        num_layer          = 10,

        max_frame_length   = 200, 
        max_target_length  = 45,
        num_target         = 59,  # including Pad token, and 'None' token
        n_epochs           = 100,
        logging_fn         = './log/training.log',
        model_fn           = './ckpt/',
        load_model_fn      = './ckpt/',

        random_seed = 42
        )
    
    pprint(dict(asdict(config).items()))

    # adj matric handling
    # if ADJ_MATRIX[0][0] != 1:
    #     ADJ_MATRIX += torch.eye(ADJ_MATRIX.shape[0])

    if config.ADJ_NORM == True:
        ADJ_MATRIX = normalizeAdjacency(ADJ_MATRIX)

    # random seed fix
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)


    with open (config.char2pred_fn, 'r') as f:
        char_to_num = json.load(f)




    # data handling
    # train_df = []
    # for train_fn in config.train_fn:
    #     train_df += [
    #         pd.read_parquet(train_fn)[['path','file_id','sequence_id','phrase']]
    #     ]
    # train_df = pd.concat(train_df)
    # train_df['y_token'] = train_df['phrase'].apply(transform_char_to_num)



    # if config.semi_preprocess_df is not None:
    #     semi_df = []
    #     for semi_fn in config.semi_preprocess_df:
    #         print(pd.read_parquet(semi_fn).tail()) 
    #         semi_df += [
    #             pd.read_parquet(semi_fn)
    #         ]
    #     semi_df = pd.concat(semi_df)
    # else:
    #     semi_df = config.semi_preprocess_df

        



    # add P token
    if config.add_padding_token == False:
        final_output = len(char_to_num)
    else:
        char_to_num['B'] = len(char_to_num)
        final_output = len(char_to_num) 

    print("-" * 100)

    # t v split  and load loader
    train_loader, valid_loader, char_to_num, num_to_char = get_loaders(
                                            config.train_fn, 
                                            char_to_num, 
                                            config, 
                                            config.frame_fn)

    # import datetime
    # start = datetime.datetime.now()
    # sample_train = next(iter(train_loader))
    # print(sample_train)
    # end = datetime.datetime.now()
    # print(end - start)
    #### one iteration took 100 seconds with batch_size 32
    sample_loader = train_loader[0]
    mini = next(iter(sample_loader))
    print("-"*100)
    print("shapes...")
    print(mini['hand_df'].shape,
          mini['adj_matrix'].shape,
          mini['y'].shape)

    print(
        '|train| =', len(sample_loader) * config.batch_size,
        '|valid| =', len(sample_loader) * config.batch_size,
    )

    n_total_iterations = len(sample_loader) * config.n_epochs
    # n_warmup_steps = int(n_total_iterations * config.warmup_ratio)
    # if config.scheduler:
    #     print(
    #         '#total_iters =', n_total_iterations,
    #         '#warmup_iters =', n_warmup_steps,
    #     )

    
    print("-"*100)
    print("Loading Model...")
    # # Load Model
    # if config.load_model_path and config.modality == 'both':
    #     model = MultiModalClassifier(config=config)                 # load frame of model
    #     package = torch.load(config.load_model_path)['model']       # load saved weight of model
    #     print(torch.load(config.load_model_path)['config'])         # check model saved path
    #     model.load_state_dict(package, strict=False)                # overide weights to model
    #     print(model)
    # else:
        # if not using multimodality or first time of fine-tunning
    


    
    model = GCNNet(
        hidden_size = mini['hand_df'].shape[-1],
        final_output=final_output,
        n_layer=config.num_layer, # 6 was shit
        activation=nn.LeakyReLU(),
    )
    print(model)



    # # set optimizer
    optimizer = get_optimizer(model, config)

    # # set loss fn

    crit = nn.CTCLoss(blank=final_output-1, zero_infinity=True)
    
    # # set warmup scheduler
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     n_warmup_steps,
    #     n_total_iterations
    # )

    # put model and lossFn on gpu
    if config.gpu_id >= 0:
        model.cuda()
        crit.cuda()

    device = next(model.parameters()).device

    min_avg_cost = 999_999_999


    # train_loader = train_loader.to(device)
    # valid_loader = valid_loader.to(device)










    ###### training, validation : add logger
    ###### save won't work -> change
    ###### DataLoader 3 iteration
    ###### preprocess 


    for epoch in range(config.n_epochs):
        for train_loader_ in train_loader: # trainloadr1
            model, train_cost = training(model, 
                                    train_loader_,
                                    optimizer,
                                    crit,
                                    epoch,
                                    ADJ_MATRIX=ADJ_MATRIX,
                                    num_to_char=num_to_char,
                                    device = device,
                                    config=config
                                    )
        total_valid_cost = 0
        for valid_loader_ in valid_loader:
            val_cost, y_s, y_hats = validating(model,
                                    valid_dataloader=valid_loader,
                                    criterion=crit,
                                    ADJ_MATRIX=ADJ_MATRIX,
                                    num_to_char=num_to_char,
                                    epoch=epoch,
                                    device=device,
                                    config=config)
            total_valid_cost += val_cost


        if min_avg_cost > val_cost:
            dict_for_infer = {
                'model':model.state_dict(),
                'config':config,
                'num_to_char':num_to_char,
                'char_to_num':char_to_num,
                'val_cost':val_cost,
                'val_y_s':y_s,
                'val_y_hats':y_hats
            }

            torch.save(dict_for_infer, f'{config.model_fn}_{epoch}_{train_cost}_{val_cost}')

            min_avg_cost = val_cost