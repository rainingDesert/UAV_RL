from ae_reward.train import train_reward, train_model, train_ae, train_gae
from ae_reward.model import AEReward
from ae_reward.autoencoder import GeneralAEModel
from ae_reward.data.dataset_param import ParamRewardDataset, ToyFullDataset
from parsers import get_train_parse

from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

import pdb

# train models
def train_aereward(args, param_key, reward_key, constraint):
    # define model
    aer_net = AEReward(gpu=args.gpu)
    if args.gpu:
        aer_net = aer_net.cuda()

    # define optimizer
    # model_opt = optim.SGD(aer_net.ae.parameters(), lr=args.model_lr, momentum=args.model_momentum)
    # reward_opt = optim.SGD(aer_net.rnet.parameters(), lr=args.reward_lr, momentum=args.reward_momentum)
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

    # train autoencoder
    pdb.set_trace()
    print('train for autoencoder-reward model')
    # freeze reward network
    for p in aer_net.rnet.parameters():
        p.requires_grad=False
        p.grad = None

    ae_ite = args.ae_ite
    for ite in tqdm(range(ae_ite)):
        # train
        train_loss_list = train_model(train_dataloader, aer_net, model_opt, args.gpu, valid=False)
        # validation
        with torch.no_grad():
            valid_loss_list = train_model(eval_dataloader, aer_net, None, args.gpu, valid=True)
            rec_loss_list = train_ae(eval_dataloader, aer_net.ae, None, args.gpu, valid=True)
        # store
        print('train loss: {}   |   valid loss: {}'.format(np.mean(train_loss_list), np.mean(valid_loss_list)))
        print('rec loss: {}'.format(np.mean(rec_loss_list)))
        train_store['train_model'].append(np.mean(train_loss_list).item())
        train_store['valid_model'].append(np.mean(valid_loss_list).item())
        train_store['valid_ae'].append(np.mean(rec_loss_list).item())

    # store
    torch.save(aer_net.cpu().state_dict(), args.model_path)
    if args.gpu:
        aer_net = aer_net.cuda()

    np.save(args.log_path, train_store)

    return aer_net

# train autoencoder only
def train_autoencoder(args, param_key, reward_key, constraint):
    # define model
    aer_net = AEReward(gpu=args.gpu)
    if args.gpu:
        aer_net = aer_net.cuda()

    # define optimizer
    # model_opt = optim.SGD(aer_net.ae.parameters(), lr=args.model_lr, momentum=args.model_momentum)
    # reward_opt = optim.SGD(aer_net.rnet.parameters(), lr=args.reward_lr, momentum=args.reward_momentum)
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
        train_loss_list = train_ae(train_dataloader, aer_net.ae, model_opt, args.gpu, valid=False)
        # validation
        with torch.no_grad():
            valid_loss_list = train_ae(eval_dataloader, aer_net.ae, None, args.gpu, valid=True)
        # store
        print('train loss: {}   |   valid loss: {}'.format(np.mean(train_loss_list), np.mean(valid_loss_list)))
        train_store['train_ae'].append(np.mean(train_loss_list).item())
        train_store['valid_ae'].append(np.mean(valid_loss_list).item())

    # store
    torch.save(aer_net.cpu().state_dict(), args.model_path)
    if args.gpu:
        aer_net = aer_net.cuda()

    np.save(args.log_path, train_store)

# test reward network
def check_rnet(args, param_key, reward_key, constraint):
    # define model
    aer_net = AEReward(gpu=args.gpu)
    if args.gpu:
        aer_net = aer_net.cuda()

    # define optimizer
    # reward_opt = optim.SGD(aer_net.rnet.parameters(), lr=args.reward_lr, momentum=args.reward_momentum)
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
    train_store = {'train_gt':[], 'train_reward':[], 'valid_gt':[], 'valid_reward':[]}

    # train reward network
    mse_loss = nn.MSELoss()
    print('train for reward network')
    re_ite = args.re_ite
    for ite in tqdm(range(re_ite)):
        # train
        train_loss_list = train_reward(train_dataloader, aer_net, reward_opt, args.gpu, valid=False)
        # validation
        aer_net.eval()
        valid_loss_list = []
        for batch_id, batch in enumerate(eval_dataloader):
            # get data
            param = batch['param']
            reward = batch['reward']
            if args.gpu:
                param = param.cuda()
                reward = reward.cuda()
            # forward and train
            with torch.no_grad():
                pred_r = aer_net.forward_reward(param)
                loss = mse_loss(pred_r.squeeze(1), reward)
            # store
            valid_loss_list.append(loss.detach().item())

        # store
        print('train loss: {}   |   valid loss: {}'.format(np.mean(train_loss_list), np.mean(valid_loss_list)))
        # train_store['train_reward'].append(np.mean(train_loss_list).item())
        # train_store['valid_reward'].append(np.mean(valid_loss_list).item())

    # visualize result
    aer_net.eval()
    for batch_id, batch in enumerate(eval_dataloader):
        # get data
        param = batch['param']
        reward = batch['reward']
        if args.gpu:
            param = param.cuda()
        # forward and train
        with torch.no_grad():
            pred_r = aer_net.forward_reward(param)
        train_store['valid_gt'] = train_store['valid_gt'] + reward.tolist()
        train_store['valid_reward'] = train_store['valid_reward'] + pred_r.squeeze(1).tolist()
    for batch_id, batch in enumerate(train_dataloader):
        # get data
        param = batch['param']
        reward = batch['reward']
        if args.gpu:
            param = param.cuda()
        # forward and train
        with torch.no_grad():
            pred_r = aer_net.forward_reward(param)
        train_store['train_gt'] = train_store['train_gt'] + reward.tolist()
        train_store['train_reward'] = train_store['train_reward'] + pred_r.squeeze(1).tolist()

    pdb.set_trace()

    np.save('store/value.npy', train_store)

    return aer_net


# test autoencoder
def check_ae(args, param_key, reward_key, constraint):
    # define model
    aer_net = AEReward(gpu=args.gpu, vae=args.vae, lat_var_dim=2)
    if args.vae:
        aer_net.load_state_dict(torch.load('store/vae.pth'))
    else:
        aer_net.load_state_dict(torch.load('store/ae.pth'))
    if args.gpu:
        aer_net = aer_net.cuda()

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
    train_store = {'train_gt':[], 'train_param':[], 'valid_gt':[], 'valid_param':[], 'valid_lat':[]}

    # visualize result
    aer_net.eval()
    for batch_id, batch in enumerate(train_dataloader):
        # get data
        param = batch['param']
        reward = batch['reward']
        if args.gpu:
            param = param.cuda()
        # forward and train
        with torch.no_grad():
            if args.vae:
                _, _, _, late_var, pred_param = aer_net.ae.forward(param)
            else:
                late_var, pred_param = aer_net.ae.forward(param)
        train_store['valid_gt'].append(param.tolist())
        train_store['valid_param'].append(pred_param.tolist())
        train_store['valid_lat'].append(late_var.tolist())

    np.save('store/param_record.npy', train_store)

    return aer_net

# test full autoencoder
def check_full_ae(args, constraint, disc_table, data_size):
    # define model
    ae_net = GeneralAEModel(gpu=args.gpu, cont_num=len(constraint), disc_num=len(disc_table), disc_table=disc_table, embed_dim=4, latent_dim=6)
    ae_net.load_state_dict(torch.load('store/full_ae.pth'))
    if args.gpu:
        ae_net = ae_net.cuda()

    # train data
    train_dataset = ToyFullDataset(constraint=constraint,
                                   disc_table=disc_table,
                                   data_size=data_size,
                                   get_ratio=0.8,
                                   train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker)

    # eval data
    eval_dataset = ToyFullDataset(constraint=constraint,
                                   disc_table=disc_table,
                                   data_size=data_size,
                                   get_ratio=0.8,
                                   train=False)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker)

    # print
    print('train dataset with size: {}   |   valid dataset with size: {}'.format(len(train_dataset), len(eval_dataset)))

    # train
    train_store = {'valid_cons_gt':[], 'valid_disc_gt':[], 'valid_cons_param':[], 'valid_disc_param':[], 'valid_lat':[]}

    # visualize result
    error_list = None
    loss_list = []
    ae_net.eval()
    for batch_id, batch in enumerate(train_dataloader):
        # get data
        cons_param = batch['cons_param'].float()
        disc_param = batch['disc_param'].long()
        if args.gpu:
            cons_param = cons_param.cuda()
            disc_param = disc_param.cuda()
        # forward and train
        with torch.no_grad():
            mse_loss, ce_loss, lat_vector, rec_cons_param, rec_disc_param = ae_net.forward(cons_param, disc_param)
            # store
            rec_disc_param = torch.cat([torch.max(per_type_disc, dim=-1, keepdims=True)[1] \
                                for per_type_disc in rec_disc_param], dim=-1)
            err = torch.sum(rec_disc_param != disc_param, dim=0)
            if error_list is None:
                error_list = err
            else:
                error_list = error_list + err
            loss_list.append(ce_loss.item())
        # store
        train_store['valid_cons_gt'].append(cons_param.tolist())
        train_store['valid_disc_gt'].append(disc_param.tolist())
        train_store['valid_cons_param'].append(rec_cons_param.tolist())
        train_store['valid_disc_param'].append(rec_disc_param.tolist())
        train_store['valid_lat'].append(lat_vector.tolist())
    
    # print error
    # pdb.set_trace()
    print('discrete parameter predict error: {}'.format((error_list/len(train_dataset)).tolist()))
    print('ce loss: {}'.format(np.mean(loss_list)))
    np.save('store/full_param_record.npy', train_store)

    return ae_net


def plot_perf(args):
    # define model
    aer_net = AEReward(gpu=args.gpu, vae=args.vae, lat_var_dim=2)
    if args.vae:
        aer_net.load_state_dict(torch.load('store/vae.pth'))
    else:
        aer_net.load_state_dict(torch.load('store/ae.pth'))
    # define reward model
    reward_network = AEReward(gpu=args.gpu, vae=args.vae, lat_var_dim=2)
    reward_network.load_state_dict(torch.load('store/after_reward.pth'))
    # cuda
    if args.gpu:
        aer_net = aer_net.cuda()
    # create latent space
    lat_vector = []
    lbs = [-1.0,-1.0]
    ubs = [1.0,1.0]
    sam_nums = 500
    for i in np.arange(lbs[0], ubs[0], (ubs[0]-lbs[0])/sam_nums):
        for j in np.arange(lbs[1], ubs[1], (ubs[1]-lbs[1])/sam_nums):
            lat_vector.append([i, j])
    # process
    with torch.no_grad():
        lat_vector = torch.tensor(lat_vector).float()
        decoded_param = aer_net.ae.forward_decode(lat_vector)
        pred_rewards = reward_network.rnet.forward(decoded_param)
    # store
    pred_rewards = pred_rewards.squeeze(1).numpy().reshape(sam_nums, sam_nums)
    X = lat_vector.numpy()[:, 0].reshape(sam_nums, sam_nums)
    Y = lat_vector.numpy()[:, 1].reshape(sam_nums, sam_nums)
    pdb.set_trace()
    result_store = [X, Y, pred_rewards]
    np.save('./store/plot_data.npy', result_store)


# do train
def do_train(constraint, param_key, reward_key):
    # define parameters
    args = get_train_parse()
    # print
    print('train reward and autoencoder...')
    print('parameter set contains: {}'.format(param_key))
    print('reward key is: {}'.format(reward_key))
    # train_aereward(args, param_key, reward_key, constraint)
    # train_autoencoder(args, param_key, reward_key, constraint)
    # check_rnet(args, param_key, reward_key, constraint)
    check_ae(args, param_key, reward_key, constraint)

# test for full
def full_test(constraint):
    # define parameters
    args = get_train_parse()
    disc_table = {0: ['TurnigyGraphene1000mAh2S75C', 'TurnigyGraphene1000mAh4S75C', 'TurnigyGraphene1300mAh4S75C', 'TurnigyGraphene1200mAh6S75C',\
                    'TurnigyGraphene1400mAh3S75C', 'TurnigyGraphene1500mAh3S75C', 'TurnigyGraphene1600mAh4S75C', 'TurnigyGraphene2200mAh3S75C'],\
                  1: ['ESC_debugging', 'kde_direct_KDEXF_UAS20LV', 'kde_direct_KDEXF_UAS55', 't_motor_AIR_20A', 't_motor_ALPHA_40A', 't_motor_AT_40A',\
                    't_motor_AT_115A'],\
                  2: ['apc_propellers_6x4EP', 'apc_propellers_6x6EP', 'apc_propellers_7x4EP', 'apc_propellers_7x5EP', 'apc_propellers_8x8EP'], \
                  3: ['t_motor_AS2312KV1400', 't_motor_AT2321KV1250', 't_motor_MN22041400KV', 't_motor_MN4010KV475', 't_motor_MT13063100KV']}
    dataset_size = 20000
    check_full_ae(args, constraint, disc_table, dataset_size)

# plot
def do_plot():
    # define parameters
    args = get_train_parse()
    # plot
    plot_perf(args)

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
    # do_train(constraint, param_key, reward_key)
    # do_plot()
    full_test(constraint)