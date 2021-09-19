from ae_reward.optimization import AEOpt, FullAEOpt, FullAEOptRun
from ae_reward.model import AEReward
from ae_reward.autoencoder import GeneralAEModel
from simulator.uav_simulation import JenkinSim, AutoTool
from LaMCTS.LA_MCTS.lamcts import MCTS
from parsers import get_opt_parse
from ae_reward.data.dataset_param import ParamRewardDataset, ToyFullDataset

from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import torch
import torch.optim as optim
import numpy as np
import random

import pdb

# optimization
def ae_optimize(param_key, constraint, jenkin_param, result_path, job_name, \
                disc_table, disc_comp, comp_ori_instance, csv_file, reward_key, gpu):
    # train data
    train_dataset = ToyFullDataset(constraint=constraint,
                                   disc_table=disc_table,
                                   data_size=300,
                                   get_ratio=1.0,
                                   train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)

    # define simulator
    params_list = []
    reward_list = []
    dims = 6
    auto_tool = AutoTool(csv_file)
    simulator = JenkinSim(param_key, constraint, jenkin_param, result_path, job_name)
    # get parameter constraint
    channel_fun = FullAEOptRun(auto_tool, simulator, reward_key, disc_table, disc_comp, comp_ori_instance, parallel_num=1)
    # simulate
    for batch_id, batch in enumerate(train_dataloader):
        print('current iteration: {}'.format(batch_id))
        # get data
        cons_param = batch['cons_param'].float()
        disc_param = batch['disc_param'].long()
        if gpu:
            cons_param = cons_param.cuda()
            disc_param = disc_param.cuda()
        # simulate
        reward = channel_fun.sim_result([cons_param[0].numpy()], [disc_param[0].numpy()], 1)
        params_list.append((cons_param[0].numpy(), disc_param[0].numpy()))
        reward_list.append(reward)
        print('discrete parameters: {}'.format(disc_param[0].numpy()))
        print('reward: {}'.format(reward))
        if batch_id != 0 and batch_id % 10 == 0:
            np.save('store/random_result.npy', {'param':params_list, 'reward':reward_list})


# do optimization
def do_opt(args, constraint, param_key, reward_key):
    # define parameters
    jenkin_param = {"graphGUID":"QuadCopter",\
                    "PETName":"/D_Testing/PET/FlightDyn_V1",\
                    "NumSamples":"1",\
                    "DesignVars":"\"Flight_Path=1,1 Analysis_Type=3,3 Length_0=200,400 Length_1=120,150 Length_8=-2,2 Length_9=-1,1 " + \
                                 "Requested_Lateral_Speed=12,12 Requested_Vertical_Speed=1,1 Q_Position=1,1 Q_Velocity=1,1 Q_Angular_velocity=1,1 Q_Angles=1,1 R=1,1\""}
    job_name = ["UAV_Workflows"]
    # discrete parameter data (0:battery, 1:ESC, 2:propeller, 3:moter)
    disc_table = {0: ['TurnigyGraphene1000mAh2S75C', 'TurnigyGraphene1000mAh4S75C', 'TurnigyGraphene1300mAh4S75C', 'TurnigyGraphene1200mAh6S75C',\
                    'TurnigyGraphene1400mAh3S75C', 'TurnigyGraphene1500mAh3S75C', 'TurnigyGraphene1600mAh4S75C', 'TurnigyGraphene2200mAh3S75C'],\
                1: ['ESC_debugging', 'kde_direct_KDEXF_UAS20LV', 'kde_direct_KDEXF_UAS55', 't_motor_AIR_20A', 't_motor_ALPHA_40A', 't_motor_AT_40A',\
                    't_motor_AT_115A'],\
                2: ['apc_propellers_6x4EP', 'apc_propellers_6x6EP', 'apc_propellers_7x4EP', 'apc_propellers_7x5EP', 'apc_propellers_8x8EP'], \
                3: ['t_motor_AS2312KV1400', 't_motor_AT2321KV1250', 't_motor_MN22041400KV', 't_motor_MN4010KV475', 't_motor_MT13063100KV']}
    disc_comp = {0:["Battery_0"], \
                 1:["ESC_0","ESC_1","ESC_2","ESC_3"], \
                 2:[["Prop_0", "Prop_2"], ["Prop_1", "Prop_3"]], \
                 3:["Motor_0", "Motor_1", "Motor_2", "Motor_3"]}
    comp_ori_instance = {"Battery_0":"TurnigyGraphene1000mAh2S75C", "ESC_0":"ESC_debugging", "ESC_1":"ESC_debugging",\
                         "ESC_2":"ESC_debugging", "ESC_3":"ESC_debugging", "Prop_0":"apc_propellers_6x4EP", "Prop_2":"apc_propellers_6x4EP",\
                         "Prop_1":"apc_propellers_6x4E", "Prop_3":"apc_propellers_6x4E", "Motor_0":"t_motor_AT2312KV1400", "Motor_1":"t_motor_AT2312KV1400",\
                         "Motor_2":"t_motor_AT2312KV1400", "Motor_3":"t_motor_AT2312KV1400"}
    disc_comp = {0:["Battery_0"], \
                 1:["ESC_0","ESC_1","ESC_2","ESC_3"], \
                 2:[["Prop_0", "Prop_2"], ["Prop_1", "Prop_3"]], \
                 3:["Motor_0", "Motor_1", "Motor_2", "Motor_3"]}
    comp_ori_instance = {"Battery_0":"TurnigyGraphene1000mAh2S75C", "ESC_0":"ESC_debugging", "ESC_1":"ESC_debugging",\
                         "ESC_2":"ESC_debugging", "ESC_3":"ESC_debugging", "Prop_0":"apc_propellers_6x4EP", "Prop_2":"apc_propellers_6x4EP",\
                         "Prop_1":"apc_propellers_6x4E", "Prop_3":"apc_propellers_6x4E", "Motor_0":"t_motor_AT2312KV1400", "Motor_1":"t_motor_AT2312KV1400",\
                         "Motor_2":"t_motor_AT2312KV1400", "Motor_3":"t_motor_AT2312KV1400"}
    args.data_path = ('D:/jwork/Agents/workspace/UAV_Workflows/results',)
    csv_file = './autograph/graphOpsUse.csv'
    # optimziation
    ae_optimize(param_key, constraint, jenkin_param, args.data_path, job_name,\
                disc_table, disc_comp, comp_ori_instance, csv_file, reward_key, args.gpu)

# execute
if __name__ == '__main__':
    # seed
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    # common parameters
    args = get_opt_parse()
    constraint = {"Length_0":(200.0, 400.0), \
                    "Length_1":(120.0,150.0), \
                    "Length_8":(-2.0, 2.0), \
                    "Length_9":(-1.0, 1.0)}
    param_key = sorted(list(constraint.keys()))
    reward_key = 'Hover_Time'
    # do optimization
    do_opt(args, constraint, param_key, reward_key)
