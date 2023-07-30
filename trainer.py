import torch
from tqdm import tqdm
import numpy as np


def training(
    model, 
    train_dataloader, 
    optimizer, 
    criterion, 
    # scaler, 
    epoch, 
    ADJ_MATRIX,
    num_to_char,
    device,
    config,
    ):

    total_cost = 0
    cost_ls = []

    model.train()
    for idx, mini in enumerate(train_dataloader):
        hand_df = mini['hand_df'].to(device)
        y = mini['y'].to(device)
        frame_length = mini['frame_length']
        y_length = mini['y_length']
        ADJ_MATRIX = ADJ_MATRIX.to(device)

        optimizer.zero_grad()

        output = model(hand_df, ADJ_MATRIX) # bs, maxlen, 59
        output = output.permute(1,0,2)  # 고쳐야하네.........ㄴㅁㄴㅇ리;ㅏㅁ너;리머ㅏㄴㅇ;ㅣ라

        torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.max_grad_norm,
            )

        loss = criterion(output, y, frame_length, y_length) # input_lengths must be of size batch_size
        # output : [200,32,59] / frame_len : 11 / y_leng : 11  -> output의 1번 사이즈와, frame사이즈가 안맞아.
        cost_ls += [loss.item()]
        total_cost += loss.item()

        loss.backward()
        optimizer.step()

        if idx%2 == 0:
            print(f"Epoch {epoch} Iteration {idx} : \
                Loss = {float(total_cost/(idx+1)):.6}, \
                    Process : {float(idx/len(train_dataloader)):.3}", 
                end = '\r')
            

        if idx%50==0:
            output = output.permute(1,0,2)
            print(f'y : {y[0].detach().cpu()}')
            print(output[0].argmax(dim=-1))
            ex_out = [num_to_char[i] for i in output[0].argmax(dim=-1).detach().cpu().numpy()]
            print(f'y_ : {ex_out}')

    return model, total_cost/len(train_dataloader)

def validating(
    model,
    valid_dataloader,
    criterion,
    ADJ_MATRIX,
    num_to_char,
    epoch,
    device,
    config,
):
    
    val_total_cost = 0
    val_cost_ls = []
    y_s = []
    y_hats = []

    model.eval()
    with torch.no_grad():
        for idx, mini in enumerate(valid_dataloader[0]):
            hand_df = mini['hand_df'].to(device)
            y = mini['y'].to(device)
            frame_length = mini['frame_length'].to(device)
            y_length = mini['y_length']
            ADJ_MATRIX = ADJ_MATRIX.to(device)

            output = model(hand_df, ADJ_MATRIX)
            output = output.permute(1,0,2)

            loss = criterion(output, y, frame_length, y_length)
            val_total_cost += loss.item()
            val_cost_ls += [loss.item()]


        if idx%2 == 0:
            print(f"Validation... Epoch {epoch} Iteration {idx} : \
                Loss = {float(val_total_cost/(idx+1)):.6},", 
                end = '\r')


        if idx%5 == 0:
            output = output.permute(1,0,2)
            y_ = y[0].detach().cpu()
            y_s += [y_]
            print(f'y : {y_}')
            ex_out = [num_to_char[i] for i in output[0].argmax(dim=-1).detach().cpu().numpy()]
            y_hats += ex_out
            print(f'y_ : {ex_out}')

    return val_total_cost/len(valid_dataloader), y_s, y_hats
    