from torch.utils.data import Dataset

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import pdb

class ParamRewardDataset(Dataset):
    # dataset with (parameters, reward)
    def __init__(self, csv_file, param_key, reward_key, constraint, get_ratio=1.0, train=True):
        # store
        self.param_key = param_key
        self.reward_key = reward_key
        self.constraint = constraint
        # load data
        df = pd.read_csv(csv_file, header=0)
        self.param_set = df[self.param_key].values
        self.reward_set = df[self.reward_key].values
        if train:
            self.param_set = self.param_set[:int(self.param_set.shape[0]*get_ratio)]
            self.reward_set = self.reward_set[:int(self.reward_set.shape[0]*get_ratio)]
        else:
            self.param_set = self.param_set[int(self.param_set.shape[0]*get_ratio):]
            self.reward_set = self.reward_set[int(self.reward_set.shape[0]*get_ratio):]
        # normalize
        self.normalize_data()

    def __len__(self):
        return self.param_set.shape[0]
    
    def __getitem__(self, idx):
        param = np.float32(self.param_set[idx])
        reward = np.float32(self.reward_set[idx])
        
        return {'param': param, 'reward':reward}
    
    def normalize_data(self):
        # check
        assert self.param_key is not None
        assert self.reward_key is not None
        assert self.constraint is not None
        # norm param
        for p_id, p_k in enumerate(self.param_key):
            cons_range = self.constraint[p_k]
            self.param_set[:, p_id] = (self.param_set[:, p_id] - cons_range[0]) / (cons_range[1] - cons_range[0])
        # map reward
        # self.reward_set = -np.log(1000/self.reward_set)
        self.reward_set = self.reward_set/100

class ToyFullDataset(Dataset):
    def __init__(self, constraint, disc_table, data_size, get_ratio=1.0, train=True):
        self.param_key = sorted(list(constraint.keys()))
        self.disc_key = sorted(list(disc_table.keys()))
        # store
        self.constraint = constraint
        self.disc_table = disc_table
        # define
        if train:
            self.data_size = int(data_size * get_ratio)
        else:
            self.data_size = data_size - int(data_size * (get_ratio))
        self.cons_set = np.random.rand(self.data_size, len(constraint))
        self.disc_set = np.random.randint([len(disc_table[d_k]) for d_k in self.disc_key], \
                                          size=(self.data_size, len(disc_table)))


    def __len__(self):
        return self.cons_set.shape[0]

    def __getitem__(self, idx):
        cons_param = self.cons_set[idx]
        disc_param = self.disc_set[idx]

        return {'cons_param':cons_param, 'disc_param':disc_param}