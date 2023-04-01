# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 17:04:29 2023

@author: TOM3O
"""
import os
import pandas as pd
import numpy as np
import random
import pickle 
#%%
def aggregate_episodes():
    #%%
    all_episodes = []
    
    file_handle_dataset = os.path.join("c:/", "Users", "TOM3O", "Desktop", "Work", "ENGG4430", "new_data.pkl")
    file_handle_dataset_old = os.path.join("c:/", "Users", "TOM3O", "Desktop", "Work", "ENGG4430", "dataset.pkl")

    if os.path.exists(file_handle_dataset) == False:
        for file in ["processed_car_following_i-80.csv"]:
            file_handle = os.path.join("c:/", "Users", "TOM3O", "Desktop", "Work", "ENGG4430", "data", file)
            
            df_chunks = pd.read_csv(file_handle, chunksize = 1000)
            df_raw =  pd.concat(df_chunks)
            
            dfs = list(df_raw.groupby(["Vehicle_ID", "Following"]))
            
            for _, event in dfs:
                
                event_length = len(event)
                num_episodes = (int)(event_length / 100)
                for i in range(num_episodes):
                    episode_raw = event[i*100:(i+1)*100]
                    dt = episode_raw["Global_Time"].diff()
                    if (len(dt.unique()) == 2) and (len(episode_raw) == 100):
                        episode = pd.DataFrame()
                        episode["Bumper to Bumper Distance"] = (episode_raw["Space_Headway_follower"] - episode_raw["v_length"]) * 0.3048
                        episode["Following Vehicle Speed"] = episode_raw["v_Vel_follower"] 
                        episode["Relative Speed"] = (episode_raw["v_Vel"] - episode_raw["v_Vel_follower"]) 
                        episode["Leading Vehicle Speed"] = episode_raw["v_Vel"] 
                        episode.reset_index(drop = True, inplace = True)
                        all_episodes.append(episode)
            
        #%%
        episodes = random.choices(all_episodes, k=1000)	
        #%%
        with open(file_handle_dataset_old, "wb") as f:
            pickle.dump(episodes, f)
            #%%
    else:
        #%%
        with open(file_handle_dataset, "rb") as f:
            episodes = pickle.load(f)
        #%%
    return episodes

#%%
# from cvxpy import *
# import numpy as np
# import scipy as sp
# import time
# import matplotlib.pyplot as plt
# import scipy.io as sio
# import os
# #%%
# root = os.path.join("c:/", "Users", "TOM3O", "Desktop", "Work", "ENGG4430", "Velocity_control")
# train_handle = os.path.join(root, 'trainSet.mat')
# test_handle = os.path.join(root, 'testSet.mat')

# # load training data
# old_train = sio.loadmat(train_handle)['calibrationData']
# old_test = sio.loadmat(test_handle)['validationData']
# trainNum = old_train.shape[0]
# testNum = old_train.shape[0]
# print('Number of training samples:', trainNum)
# print('Number of validate samples:', testNum)
# #%%
# episode_dfs = []
# for episode_raw in train:
#     episode_dfs.append(pd.DataFrame(data = episode_raw[0], columns = ["Space,", "speed", "rel speed", "leading vehicle speed"]))
