import os
import numpy as np
import pandas as pd
from math import floor
from pathlib import Path
from collections import OrderedDict

horizon_k = [10, 20, 50, 100]
jump_threshold = 0.2
alpha = 0.001

def label_processing(source_path, save_path):
    print(f"Processing: {source_path}, Saving: {save_path}")
    data = np.loadtxt(source_path)
    first_price = data[0,:]
    jump_points = [i for i in range(1, len(first_price)) if abs(first_price[i-1] - first_price[i]) > jump_threshold]

    left_bound = 0
    series_list = []
    for i in range(5):
        right_bound = jump_points[i] if i <= 3 else len(first_price)
        stock_data = data[:, left_bound:right_bound]
        series_list.append(stock_data[:40,:])
        left_bound = right_bound

    processed_stock_data = OrderedDict()
    all_stock_df = None
    for s in series_list:  # s.shape = (40, T), five stocks
        temp_df = pd.DataFrame()
        temp_df['mid_price'] = (s[0,:] + s[2,:])/2
        for k in horizon_k:
            temp_df['mm_'+str(k)] = temp_df['mid_price']  # mean of previous k ticks
            temp_df['mp_'+str(k)] = temp_df['mid_price']  # mean of next k ticks 
            for r in range(k):
                temp_df['mm_'+str(k)] += temp_df['mid_price'].shift(periods=(r+1)).fillna(0)
                temp_df['mp_'+str(k)] += temp_df['mid_price'].shift(periods=-1*(r+1)).fillna(0)

            temp_df['mm_'+str(k)] = temp_df['mm_'+str(k)]/k
            temp_df['mp_'+str(k)] = temp_df['mp_'+str(k)]/k
    
        temp_df = temp_df[max(horizon_k):-1*max(horizon_k)]
        ranges = [-10**6, -1*alpha, alpha, 10**6 ]  # alpha = 0.001

        for k in horizon_k:
            lt_k = (temp_df['mp_'+str(k)] - temp_df['mm_'+str(k)])/temp_df['mm_'+str(k)].abs()
            temp_df['label_'+str(k)] = pd.cut(lt_k, ranges, right = False, labels = ['-1', '0', '1'])
    
        
        for i in range(0, s.shape[0]):
            if i % 4 == 0:
                processed_stock_data[f'p_ask_{floor(i/4)}'] = s[i,max(horizon_k):-1*max(horizon_k)]
            elif i % 4 == 1:
                processed_stock_data[f'v_ask_{floor(i/4)}'] = s[i,max(horizon_k):-1*max(horizon_k)]
            elif i % 4 == 2:
                processed_stock_data[f'p_bid_{floor(i/4)}'] = s[i,max(horizon_k):-1*max(horizon_k)]
            else:
                processed_stock_data[f'v_bid_{floor(i/4)}'] = s[i,max(horizon_k):-1*max(horizon_k)]
    
        for k in horizon_k:
            processed_stock_data[f'label_{k}'] = temp_df['label_'+str(k)].astype(float).values + 2
    
        one_stock_df = pd.DataFrame(processed_stock_data)
        if all_stock_df is None:
            all_stock_df = one_stock_df
        else:
            all_stock_df = pd.concat([all_stock_df, one_stock_df], axis=0)
    
    print(f"data.shape: {data.shape}, stock_df.values.shape: {all_stock_df.values.shape}")
    np.savetxt(save_path, np.transpose(all_stock_df.values))

if __name__ == '__main__':
    raw_input_dir = 'zscore'
    saved_input_dir = 'processed_zscore'
    split_dir = ['training', 'testing']
    for split in split_dir:
        raw_file_dir = f'{raw_input_dir}/{split}'
        Path(f"{saved_input_dir}/{split}").mkdir(parents=True, exist_ok=True)
        raw_txt_files = os.listdir(raw_file_dir)
        for raw_file in raw_txt_files: # *.txt
            saved_path = f'{saved_input_dir}/{split}/{raw_file}'
            label_processing(os.path.join(raw_file_dir,raw_file), saved_path)