import json
import pandas as pd,numpy as np,os
import json
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
from pathlib import Path
import os

print("importing..")


def preprocess_frames(df):

    check_na = pd.DataFrame(df.isna().sum(axis=1))
    check_na[1] = check_na[0].shift(1)
    check_na['remove'] = (check_na[0] == check_na[1]) & (check_na[0] != 0)
    final_df = df.drop(check_na[check_na['remove']].index, axis=0).fillna(0)
    return final_df


with open ("./data/character_to_prediction_index.json", 'r') as f:
    char_to_num = json.load(f)


# Load the supplemental_metadata.csv file into memory
supplemental_df = pd.read_csv("./data/supplemental_metadata.csv")
train_df = pd.read_csv("./data/train.csv")


# make variable
train_df['y_len'] = train_df['phrase'].apply(len)
supplemental_df['y_len'] = supplemental_df['phrase'].apply(len)



# train, x,y,z 좌표들만 모아서 ls에 붙여넣기
ls = []
# for i in os.listdir('./data/train_landmarks/'):
#     tmp_df = pd.read_parquet(f"./data/train_landmarks/{i}").reset_index()

#     x_right_hand = [f'x_right_hand_{i}' for i in range(21)]
#     y_right_hand = [f'y_right_hand_{i}' for i in range(21)]
#     z_right_hand = [f'z_right_hand_{i}' for i in range(21)]

#     x_left_hand = [f'x_left_hand_{i}' for i in range(21)]
#     y_left_hand = [f'y_left_hand_{i}' for i in range(21)]
#     z_left_hand = [f'z_left_hand_{i}' for i in range(21)]

#     ls += [tmp_df[['sequence_id','frame']+x_right_hand + y_right_hand + z_right_hand + x_left_hand + y_left_hand + z_left_hand]]

# partial_train_landmark_df = pd.concat(ls)
# partial_train_landmark_df.to_parquet("./data/partial_train_landmark01.parquet")


# # frame길이와, y_len길이 비교해서   frame이 긴 row들만 남기기
# sequence_df = partial_train_landmark_df.set_index("sequence_id")

# result = []

# for idx, row in train_df.iterrows():
#     tmp_sequence_df = sequence_df.loc[row['sequence_id']]

#     x_right_hand = [f'x_right_hand_{i}' for i in range(21)]
#     y_right_hand = [f'y_right_hand_{i}' for i in range(21)]
#     z_right_hand = [f'z_right_hand_{i}' for i in range(21)]

#     x_left_hand = [f'x_left_hand_{i}' for i in range(21)]
#     y_left_hand = [f'y_left_hand_{i}' for i in range(21)]
#     z_left_hand = [f'z_left_hand_{i}' for i in range(21)]

#     right_df = tmp_sequence_df[x_right_hand + y_right_hand + z_right_hand]
#     right_df = right_df.reset_index()
#     right_final_df = preprocess_frames(right_df)

#     left_df = tmp_sequence_df[x_left_hand + y_left_hand + z_left_hand]
#     left_df = left_df.reset_index()
#     left_final_df = preprocess_frames(left_df)

#     result += [{'idx':idx, 'right_len':len(right_final_df), 'left_len':len(left_final_df)}]

# result_df = pd.DataFrame(result)
# result_df.to_parquet("./data/partial_train_landmark02.parquet")

# train_df1 = train_df.reset_index().rename(columns = {'index':"idx"}).merge(result_df, on='idx', how='left')
# train_df1['frame_len'] = train_df1[['right_len','left_len']].max(axis=1)
# train_df1[train_df1['y_len'] <= train_df1['frame_len']].to_parquet("./data/train_possible.parquet")


# ###################

ls = []
for i in os.listdir('./data/supplemental_landmarks/'):
    tmp_df = pd.read_parquet(f"./data/supplemental_landmarks/{i}").reset_index()

    x_right_hand = [f'x_right_hand_{i}' for i in range(21)]
    y_right_hand = [f'y_right_hand_{i}' for i in range(21)]
    z_right_hand = [f'z_right_hand_{i}' for i in range(21)]

    x_left_hand = [f'x_left_hand_{i}' for i in range(21)]
    y_left_hand = [f'y_left_hand_{i}' for i in range(21)]
    z_left_hand = [f'z_left_hand_{i}' for i in range(21)]

    ls += [tmp_df[['sequence_id','frame']+x_right_hand + y_right_hand + z_right_hand + x_left_hand + y_left_hand + z_left_hand]]
    

partial_supplemental_df = pd.concat(ls)
partial_supplemental_df.to_parquet("./data/partial_supplemental_landmark01.parquet")
print('ok'*100)

# frame길이와 y_len길이 비교해서 ~~~
sequence_df = partial_supplemental_df.set_index("sequence_id")

result = []
except_ls = []
for idx, row in supplemental_df.iterrows():
    try:
        tmp_sequence_df = sequence_df.loc[row['sequence_id']]
    except:
        except_ls += [row['sequence_id']]
        continue
    x_right_hand = [f'x_right_hand_{i}' for i in range(21)]
    y_right_hand = [f'y_right_hand_{i}' for i in range(21)]
    z_right_hand = [f'z_right_hand_{i}' for i in range(21)]

    x_left_hand = [f'x_left_hand_{i}' for i in range(21)]
    y_left_hand = [f'y_left_hand_{i}' for i in range(21)]
    z_left_hand = [f'z_left_hand_{i}' for i in range(21)]

    right_df = tmp_sequence_df[x_right_hand + y_right_hand + z_right_hand]
    right_df = right_df.reset_index()
    right_final_df = preprocess_frames(right_df)

    left_df = tmp_sequence_df[x_left_hand + y_left_hand + z_left_hand]
    left_df = left_df.reset_index()
    left_final_df = preprocess_frames(left_df)

    result += [{'idx':idx, 'right_len':len(right_final_df), 'left_len':len(left_final_df)}]


result_sup = pd.DataFrame(result)
result_sup.to_parquet("./data/partial_supplemental_landmark02.parquet")

# sequence_df1 = pd.concat([partial_supplemental_df, result_sup])
supplemental_df1 = supplemental_df.reset_index().rename(columns = {'index':"idx"}).merge(result_sup, on='idx', how='left')
supplemental_df1['frame_len'] = supplemental_df1[['right_len','left_len']].max(axis=1)
supplemental_df1[supplemental_df1['y_len'] <= supplemental_df1['frame_len']].to_parquet("./data/supplemental_possible.parquet")