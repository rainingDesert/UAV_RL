from ae_reward.train import train_reward, train_model, train_ae
from ae_reward.model import AEReward
from ae_reward.data.dataset_param import ParamRewardDataset
from parsers import get_train_parse

from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import numpy as np
import random

import pdb

# train models
def train_rewardnet(args, param_key, reward_key, constraint):
    # define model
    aer_net = AEReward(gpu=args.gpu, vae=False, lat_var_dim=2)
    if args.gpu:
        aer_net = aer_net.cuda()

    # define optimizer
    model_opt = optim.Adam(aer_net.ae.parameters(), lr=args.model_lr)
    reward_opt = optim.Adam(aer_net.rnet.parameters(), lr=args.reward_lr)

    # train data
    train_dataset = ParamRewardDataset(csv_file=args.csv_file, 
                                 param_key=param_key, 
                                 reward_key=reward_key, 
                                 constraint=constraint,
                                 get_ratio = args.train_ratio,
                                 train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker)

    # eval data
    eval_dataset = ParamRewardDataset(csv_file=args.csv_file, 
                                 param_key=param_key, 
                                 reward_key=reward_key, 
                                 constraint=constraint,
                                 get_ratio = args.train_ratio,
                                 train=False)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker)

    # print
    print('train dataset with size: {}   |   valid dataset with size: {}'.format(len(train_dataset), len(eval_dataset)))

    # train
    train_store = {'train_reward':[], 'valid_reward':[], 'train_model':[], 'valid_model':[], 'valid_ae':[]}

    # train reward network
    print('train for reward network')
    re_ite = args.re_ite
    for ite in tqdm(range(re_ite)):
        # train
        train_loss_list = train_reward(train_dataloader, aer_net, reward_opt, args.gpu, valid=False)
        # validation
        with torch.no_grad():
            valid_loss_list = train_reward(eval_dataloader, aer_net, None, args.gpu, valid=True)
        # store
        print('train loss: {}   |   valid loss: {}'.format(np.mean(train_loss_list), np.mean(valid_loss_list)))
        train_store['train_reward'].append(np.mean(train_loss_list).item())
        train_store['valid_reward'].append(np.mean(valid_loss_list).item())

    # store
    torch.save(aer_net.cpu().state_dict(), args.reward_path)
    if args.gpu:
        aer_net = aer_net.cuda()

    np.save(args.reward_log_path, train_store)

    return aer_net

# train autoencoder only
def train_autoencoder(args, param_key, reward_key, constraint):
    # define model
    aer_net = AEReward(gpu=args.gpu, vae=args.vae, lat_var_dim=2)
    if args.gpu:
        aer_net = aer_net.cuda()

    # define optimizer
    model_opt = optim.Adam(aer_net.ae.parameters(), lr=args.model_lr)

    # train data
    train_dataset = ParamRewardDataset(csv_file=args.csv_file, 
                                 param_key=param_key, 
                                 reward_key=reward_key, 
                                 constraint=constraint,
                                 get_ratio = args.train_ratio,
                                 train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker)

    # eval data
    eval_dataset = ParamRewardDataset(csv_file=args.csv_file, 
                                 param_key=param_key, 
                                 reward_key=reward_key, 
                                 constraint=constraint,
                                 get_ratio = args.train_ratio,
                                 train=False)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker)

    # print
    print('train dataset with size: {}   |   valid dataset with size: {}'.format(len(train_dataset), len(eval_dataset)))

    # train
    train_store = {'train_ae':[], 'valid_ae':[]}

    # train reward network
    print('train for reward network')
    ae_ite = args.ae_ite
    for ite in tqdm(range(ae_ite)):
        # train
        train_loss_list = train_ae(train_dataloader, aer_net.ae, model_opt, args.gpu, valid=False, vae=args.vae)
        # validation
        with torch.no_grad():
            valid_loss_list = train_ae(eval_dataloader, aer_net.ae, None, args.gpu, valid=True, vae=args.vae)
        # store
        if args.vae:
            print('train loss: {}   |   valid loss: {}   |   beta: {}'.format(np.mean(np.asarray(train_loss_list)[:, 0]), \
                                                                              np.mean(np.asarray(valid_loss_list)[:, 0]), \
                                                                              aer_net.ae.beta))
            print('train rec loss: {}   |   valid rec loss: {}'.format(np.mean(np.asarray(train_loss_list)[:, 1]), np.mean(np.asarray(valid_loss_list)[:, 1])))
            print('train kl loss: {}   |   valid kl loss: {}'.format(np.mean(np.asarray(train_loss_list)[:, 2]), np.mean(np.asarray(valid_loss_list)[:, 2])))
            print('----------------------------------------------')
            train_store['train_ae'].append(np.mean(np.asarray(train_loss_list), axis=0))
            train_store['valid_ae'].append(np.mean(np.asarray(valid_loss_list), axis=0))
        else:
            print('train loss: {}   |   valid loss: {}'.format(np.mean(train_loss_list), np.mean(valid_loss_list)))
            train_store['train_ae'].append(np.mean(train_loss_list).item())
            train_store['valid_ae'].append(np.mean(valid_loss_list).item())

    # store
    if args.vae:
        print('final beta is: {}'.format(aer_net.ae.beta))
    torch.save(aer_net.cpu().state_dict(), args.model_path)
    if args.gpu:
        aer_net = aer_net.cuda()

    np.save(args.log_path, train_store)

# do train
def do_train(constraint, param_key, reward_key):
    # define parameters
    args = get_train_parse()
    # print
    print('train reward and autoencoder...')
    print('parameter set contains: {}'.format(param_key))
    print('reward key is: {}'.format(reward_key))
    train_rewardnet(args, param_key, reward_key, constraint)
    train_autoencoder(args, param_key, reward_key, constraint)


# execute
if __name__ == '__main__':
    # seed
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    # common parameters
    constraint = {"ArmLength":(200.0, 400.0), \
                    "SupportLength":(120.0,150.0), \
                    "BatteryOffset_X":(-2.0, 2.0), \
                    "BatteryOffset_z":(-1.0, 1.0)}
    param_key = sorted(list(constraint.keys()))
    reward_key = 'Max_Hover_Time__s'
    # do train
    do_train(constraint, param_key, reward_key)

