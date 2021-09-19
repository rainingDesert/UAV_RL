from ae_reward.train import train_reward, train_model, train_ae, train_gae
from ae_reward.model import AEReward
from ae_reward.autoencoder import GeneralAEModel
from ae_reward.data.dataset_param import ParamRewardDataset, ToyFullDataset
from parsers import get_train_parse

from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import numpy as np
import random

import pdb

# train general autoencoder
def train_general_ae(args, constraint, disc_table, data_size):
    # define model
    ae_net = GeneralAEModel(gpu=args.gpu, cont_num=len(constraint), disc_num=len(disc_table), disc_table=disc_table, embed_dim=4, latent_dim=6)
    if args.gpu:
        ae_net = ae_net.cuda()

    # define optimizer
    model_opt = optim.Adam(ae_net.parameters(), lr=args.model_lr)

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
    train_store = {'train_ae':[], 'valid_ae':[]}

    # train autoencoder
    print('train for full autoencoder')
    ae_ite = args.ae_ite
    for ite in tqdm(range(ae_ite)):
        # train
        train_loss_list = train_gae(train_dataloader, ae_net, model_opt, args.gpu, valid=False, vae=args.vae)
        # pdb.set_trace()
        # validation
        with torch.no_grad():
            valid_loss_list = train_gae(eval_dataloader, ae_net, None, args.gpu, valid=True, vae=args.vae)
        # store
        print('train loss: {}   |   valid loss: {}'.format(np.mean(np.asarray(train_loss_list)[:, 0]), \
                                                                            np.mean(np.asarray(valid_loss_list)[:, 0])))
        print('train rec continuous loss: {}   |   valid rec continuous loss: {}'.format(np.mean(np.asarray(train_loss_list)[:, 1]), np.mean(np.asarray(valid_loss_list)[:, 1])))
        print('train rec discrete loss: {}   |   valid rec discrete loss: {}'.format(np.mean(np.asarray(train_loss_list)[:, 2]), np.mean(np.asarray(valid_loss_list)[:, 2])))
        print('----------------------------------------------')
        train_store['train_ae'].append(np.mean(np.asarray(train_loss_list), axis=0))
        train_store['valid_ae'].append(np.mean(np.asarray(valid_loss_list), axis=0))
    # store
    torch.save(ae_net.cpu().state_dict(), args.model_path)
    if args.gpu:
        ae_net = ae_net.cuda()

    np.save(args.log_path, train_store)

# do train
def do_train(constraint, disc_table):
    # define parameters
    args = get_train_parse()
    param_key = sorted(list(constraint.keys()))
    data_size = 20000
    # print
    print('train autoencoder...')
    print('parameter set contains: {}'.format(param_key))
    train_general_ae(args, constraint, disc_table, data_size)


# execute
if __name__ == '__main__':
    # seed
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    # common parameters
    constraint = {"Length_0":(200.0, 400.0), \
                    "Length_1":(120.0,150.0), \
                    "Length_8":(-2.0, 2.0), \
                    "Length_9":(-1.0, 1.0)}
    # discrete parameter data (0:battery, 1:ESC, 2:propeller, 3:moter)
    disc_table = {0: ['TurnigyGraphene1000mAh2S75C', 'TurnigyGraphene1000mAh4S75C', 'TurnigyGraphene1300mAh4S75C', 'TurnigyGraphene1200mAh6S75C',\
                      'TurnigyGraphene1400mAh3S75C', 'TurnigyGraphene1500mAh3S75C', 'TurnigyGraphene1600mAh4S75C', 'TurnigyGraphene2200mAh3S75C'],\
                  1: ['ESC_debugging', 'kde_direct_KDEXF_UAS20LV', 'kde_direct_KDEXF_UAS55', 't_motor_AIR_20A', 't_motor_ALPHA_40A', 't_motor_AT_40A',\
                      't_motor_AT_115A'],\
                  2: ['apc_propellers_6x4EP', 'apc_propellers_6x6EP', 'apc_propellers_7x4EP', 'apc_propellers_7x5EP', 'apc_propellers_8x8EP'], \
                  3: ['t_motor_AS2312KV1400', 't_motor_AT2321KV1250', 't_motor_MN22041400KV', 't_motor_MN4010KV475', 't_motor_MT13063100KV']}
    # do train
    do_train(constraint, disc_table)

