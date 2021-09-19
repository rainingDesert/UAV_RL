from lstm_rl.model import Controller
from lstm_rl.data.data_check import DataCheck
from simulator.uav_simulation import JenkinSim, AutoTool
from ae_reward.optimization import FullAEOpt, FullAEOptRun
from parsers import get_rl_parse

from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import numpy as np
import random

import pdb

# train controller by reinforcement
def train_general_ae(args, simulator, constraint, disc_table):
    # define model
    controller = Controller(embed_dim=4, hidden_dim=10, lstm_layer=2, cont_num=len(constraint), disc_table=disc_table, gpu=args.gpu, var_init=0.1)
    if args.gpu:
        controller = controller.cuda()
    
    # load
    if args.load_path is not None:
        print('log controller from {}'.format(args.load_path))
        controller.load_state_dict(torch.load(args.load_path))

    # define optimizer
    model_opt = optim.Adam(controller.parameters(), lr=args.lr)

    # data checker
    data_checker = DataCheck(args.csv_file)

    # print
    print('checker contains {} history datas'.format(len(data_checker)))

    loss_store = {'reward':[], 'loss':[], 'logits_check':[], 'real_time_param':[], 'real_time_reward':[], 'baseline':[], 'logstd':[]}
    # train rl
    for ite in range(args.iteration):
        print('----------- current iteration: {} ------------------'.format(ite))
        disc_result, probs_result, reg_result, sample_result, logits_check = controller.forward(args.sample_num)
        # simulate
        reward_disc = {}
        new_disc_result = []
        new_sample_result = []
        # check
        for param_id in range(disc_result.shape[0]):
            print('discrete parameter {}: {}'.format(param_id, disc_result[param_id].tolist()))
            print('continuous parameter {}: {}'.format(param_id, sample_result[param_id].tolist()))
            # check valid
            check_data_dict = {key_id:disc_result[param_id][key_id].item() for key_id in range(disc_result[param_id].shape[0])}
            valid = data_checker.check(check_data_dict)
            if valid == 0:
                reward_disc[param_id] = -1.0
                print('get nonvalid reward')
            else:
                new_disc_result.append(disc_result[param_id].tolist())
                new_sample_result.append(sample_result[param_id].tolist())
            print('-----------------------')
        print()
        # execute
        all_reward_list = []
        for exec_id in range(0, len(new_disc_result), args.parallel_num):
            # print
            print('result for execute: {}'.format(exec_id))
            total_num = min(len(new_disc_result)-exec_id, args.parallel_num)
            for param_id in range(total_num):
                cur_id = exec_id+param_id
                print('discrete parameter {}: {}'.format(cur_id, new_disc_result[cur_id]))
                print('continuous parameter {}: {}'.format(cur_id, new_sample_result[cur_id]))
            # execute
            reward_list = simulator.sim_result(np.asarray(new_sample_result[exec_id:exec_id+total_num]),
                                               np.asarray(new_disc_result[exec_id:exec_id+total_num]),
                                               run_parallel=total_num)
            process_reward_list = []
            if total_num == 1:
                reward_list = [reward_list]
            for r_id, r in enumerate(reward_list):
                if r == 0.0:
                    process_reward_list.append(-1.0)
                else:
                    process_reward_list.append(r/100.0)
                # store into dataloader
                cur_id = exec_id+r_id
                check_data_dict = {str(key_id):new_disc_result[cur_id][key_id] for key_id in range(len(new_disc_result[cur_id]))}
                data_checker.add_data(check_data_dict, r!=0.0)
            all_reward_list += process_reward_list
            # print
            print('reward: {}'.format(process_reward_list))
            print('-------------------------------')
        # combine
        invalid_id = sorted(list(reward_disc.keys()))
        for each_id in invalid_id:
            all_reward_list.insert(each_id, reward_disc[each_id])
        # get losss
        reward_list = torch.tensor(all_reward_list)
        # reward_list = torch.tensor([0.156,1.21,2.51,1.265])
        loss = controller.cal_loss(probs_result, reg_result, sample_result, reward_list)
        # backprop
        model_opt.zero_grad()
        loss.backward()
        model_opt.step()
        # store
        loss_store['reward'].append(torch.mean(reward_list).item())
        loss_store['loss'].append(loss.detach().item())
        loss_store['baseline'].append(controller.baseline.item())
        loss_store['logits_check'].append(logits_check.tolist())
        loss_store['logstd'].append(controller.logstd.tolist())
        for param_id in range(disc_result.shape[0]):
            loss_store['real_time_param'].append([sample_result[param_id].detach().tolist(), disc_result[param_id].detach().tolist()])
            loss_store['real_time_reward'].append(reward_list[param_id])
        print('current logits check: {}'.format(logits_check.tolist()))
        print('current std: {}'.format(torch.exp(controller.logstd).tolist()))
        print('current reward: {}, current loss: {}'.format(torch.mean(reward_list).item(), loss.detach().item()))
        if ite % 5 == 0:
            np.save(args.log_path, loss_store)
            data_checker.store_data(args.new_csv_file)
        # store model
        torch.save(controller.cpu().state_dict(), args.model_path)
        if args.gpu:
            controller = controller.cuda()

    # store
    torch.save(controller.cpu().state_dict(), args.model_path)
    if args.gpu:
        controller = controller.cuda()

# do train
def do_train(constraint, disc_table):
    # define parameters
    args = get_rl_parse()
    param_key = sorted(list(constraint.keys()))
    # print
    print('continuous parameter set contains: {}'.format(param_key))
    # get simulator
    jenkin_param = {"graphGUID":"QuadCopter",\
                "PETName":"/D_Testing/PET/FlightDyn_V1",\
                "NumSamples":"1",\
                "DesignVars":"\"Flight_Path=1,1 Analysis_Type=3,3 Length_0=200,400 Length_1=120,150 Length_8=-2,2 Length_9=-1,1 " + \
                                "Requested_Lateral_Speed=12,12 Requested_Vertical_Speed=1,1 Q_Position=1,1 Q_Velocity=1,1 Q_Angular_velocity=1,1 Q_Angles=1,1 R=1,1\""}
    disc_comp = {0:["Battery_0"], \
                 1:["ESC_0","ESC_1","ESC_2","ESC_3"], \
                 2:[["Prop_0", "Prop_2"], ["Prop_1", "Prop_3"]], \
                 3:["Motor_0", "Motor_1", "Motor_2", "Motor_3"]}
    comp_ori_instance = {"Battery_0":"TurnigyGraphene1000mAh2S75C", "ESC_0":"ESC_debugging", "ESC_1":"ESC_debugging",\
                         "ESC_2":"ESC_debugging", "ESC_3":"ESC_debugging", "Prop_0":"apc_propellers_6x4EP", "Prop_2":"apc_propellers_6x4EP",\
                         "Prop_1":"apc_propellers_6x4E", "Prop_3":"apc_propellers_6x4E", "Motor_0":"t_motor_AT2312KV1400", "Motor_1":"t_motor_AT2312KV1400",\
                         "Motor_2":"t_motor_AT2312KV1400", "Motor_3":"t_motor_AT2312KV1400"}
    job_name = ("UAV_Workflows", "UAV_Workflows_2", "UAV_Workflows_3", "UAV_Workflows_4")
    reward_key = 'Hover_Time'
    dims = 6
    store_path = 'store'
    # create jenkin simulation object
    auto_tool = AutoTool('./autograph/graphOpsUse.csv')
    args.data_path = ('D:/jwork/Agents/workspace/UAV_Workflows/results', 'D:/jwork/Agents/workspace/UAV_Workflows_2/results', \
                      'D:/jwork/Agents/workspace/UAV_Workflows_3/results', 'D:/jwork/Agents/workspace/UAV_Workflows_4/results')
    jenkin_sim = JenkinSim(args.url, args.usr_name, args.password, param_key, constraint, jenkin_param, args.data_path, job_name)
    # get parameter constraint
    simulator = FullAEOptRun(auto_tool, jenkin_sim, reward_key, disc_table, disc_comp, comp_ori_instance, args.parallel_num)
    train_general_ae(args, simulator, constraint, disc_table)


# execute
if __name__ == '__main__':
    # seed
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    # common parameters
    constraint = {"Length_0":(200.0, 400.0), \
                    "Length_1":(120.0, 150.0), \
                    "Length_8":(-2.0, 2.0), \
                    "Length_9":(-2.0, 2.0)}
    # discrete parameter data (0:battery, 1:ESC, 2:propeller, 3:moter)
    disc_table = {0: ['TurnigyGraphene1000mAh2S75C', 'TurnigyGraphene1000mAh3S75C', 'TurnigyGraphene1000mAh4S75C', 'TurnigyGraphene1000mAh6S75C', \
                      'TurnigyGraphene1200mAh6S75C', 'TurnigyGraphene1300mAh3S75C', 'TurnigyGraphene1300mAh4S75C', 'TurnigyGraphene1400mAh3S75C', \
                      'TurnigyGraphene1400mAh4S75C', 'TurnigyGraphene1500mAh3S75C', 'TurnigyGraphene1500mAh4S75C', 'TurnigyGraphene1600mAh4S75C', \
                      'TurnigyGraphene1600mAh4S75CSquare', 'TurnigyGraphene2200mAh3S75C', 'TurnigyGraphene2200mAh4S75C', 'TurnigyGraphene3000mAh3S75C', \
                      'TurnigyGraphene3000mAh4S75C', 'TurnigyGraphene3000mAh6S75C', 'TurnigyGraphene4000mAh3S75C', 'TurnigyGraphene4000mAh4S75C', \
                      'TurnigyGraphene4000mAh6S75C', 'TurnigyGraphene5000mAh3S75C', 'TurnigyGraphene5000mAh4S75C', 'TurnigyGraphene5000mAh6S75C', \
                      'TurnigyGraphene6000mAh3S75C', 'TurnigyGraphene6000mAh4S75C', 'TurnigyGraphene6000mAh6S75C'],\
                  1: ['ESC_debugging', 'kde_direct_KDEXF_UAS20LV', 'kde_direct_KDEXF_UAS35', 'kde_direct_KDEXF_UAS55', 't_motor_AIR_20A', 't_motor_AIR_40A',\
                      't_motor_ALPHA_40A', 't_motor_ALPHA_60A', 't_motor_AT_115A', 't_motor_AT_12A', 't_motor_AT_20A', 't_motor_AT_30A', 't_motor_AT_40A', \
                      't_motor_AT_55A', 't_motor_AT_75A', 't_motor_FLAME_100A', 't_motor_FLAME_60A', 't_motor_FLAME_70A', 't_motor_T_60A', 't_motor_T_80A'],\
                  2: ['apc_propellers_6x4EP', 'apc_propellers_10x5EP', 'apc_propellers_10x6EP', 'apc_propellers_10x7EP', 'apc_propellers_11x5_5EP', \
                      'apc_propellers_11x8EP', 'apc_propellers_12x12EP', 'apc_propellers_12x6EP', 'apc_propellers_12x8EP', 'apc_propellers_13x10EP', \
                      'apc_propellers_13x4EP', 'apc_propellers_13x5_5EP', 'apc_propellers_13x6_5EP', 'apc_propellers_13x8EP', 'apc_propellers_14x7EP', \
                      'apc_propellers_14x8_5EP', 'apc_propellers_15x10EP', 'apc_propellers_15x4EP', 'apc_propellers_16x10EP', 'apc_propellers_16x4EP', \
                      'apc_propellers_18x10EP', 'apc_propellers_18x8EP', 'apc_propellers_19x10EP', 'apc_propellers_20x10EP', 'apc_propellers_20x13EP', \
                      'apc_propellers_27x13EP', 'apc_propellers_4_1x4_1EP', 'apc_propellers_4_75x4_75EP', 'apc_propellers_5_5x4_5EP', 'apc_propellers_5x3EP', \
                      'apc_propellers_5x5EP', 'apc_propellers_5x7_5EP', 'apc_propellers_6x4EP', 'apc_propellers_6x6EP', 'apc_propellers_7x4EP', 'apc_propellers_7x5EP', \
                      'apc_propellers_7x6EP', 'apc_propellers_8x6EP', 'apc_propellers_8x8EP', 'apc_propellers_9x4_5EP', 'apc_propellers_9x6EP'], \
                  3: ['kde_direct_KDE2306XF2550', 'kde_direct_KDE2315XF885', 'kde_direct_KDE2315XF965', 'kde_direct_KDE2814XF_515', 'kde_direct_KDE2814XF_775', \
                      'kde_direct_KDE3510XF_475', 'kde_direct_KDE3510XF_715', 'kde_direct_KDE3520XF_400', 'kde_direct_KDE4012XF_400', 'kde_direct_KDE4014XF_380', \
                      'kde_direct_KDE4213XF_360', 't_motor_AS2308KV1450', 't_motor_AS2308KV2600', 't_motor_AS2312KV1150', 't_motor_AS2312KV1400', 't_motor_AS2317KV1250',\
                      't_motor_AS2317KV1400', 't_motor_AS2317KV880', 't_motor_AS2814KV1050', 't_motor_AS2814KV1200', 't_motor_AS2814KV2000', 't_motor_AS2814KV900',\
                      't_motor_AS2820KV1050', 't_motor_AS2820KV1250', 't_motor_AS2820KV880', 't_motor_AT2308KV1450', 't_motor_AT2308KV2600', 't_motor_AT2310KV2200',\
                      't_motor_AT2312KV1150', 't_motor_AT2312KV1400', 't_motor_AT2317KV1250', 't_motor_AT2317KV1400', 't_motor_AT2317KV880', 't_motor_AT2321KV1250', \
                      't_motor_AT2321KV950', 't_motor_AT2814KV1050', 't_motor_AT2814KV1200', 't_motor_AT2814KV900', 't_motor_AT2820KV1050', 't_motor_AT2820KV1250',\
                      't_motor_AT2820KV880', 't_motor_AT2826KV1100', 't_motor_AT2826KV900', 't_motor_AT3520KV550', 't_motor_AT3520KV720', 't_motor_AT3520KV850', \
                      't_motor_AT3530KV580', 't_motor_AT4120KV250', 't_motor_AT4120KV500', 't_motor_AT4120KV560', 't_motor_AT4125KV250', 't_motor_AT4125KV540', \
                      't_motor_AT4130KV230', 't_motor_AT4130KV300', 't_motor_AT4130KV450', 't_motor_MN22041400KV', 't_motor_MN2212KV780', 't_motor_MN2212KV920', \
                      't_motor_MN3110KV470', 't_motor_MN3110KV700', 't_motor_MN3110KV780', 't_motor_MN3508KV380', 't_motor_MN3508KV580', 't_motor_MN3508KV700', \
                      't_motor_MN3510KV360', 't_motor_MN3510KV630', 't_motor_MN3510KV700', 't_motor_MN3515KV400', 't_motor_MN3520KV400', 't_motor_MN4010KV370', \
                      't_motor_MN4010KV475', 't_motor_MN4010KV580', 't_motor_MN4012KV340', 't_motor_MN4012KV400', 't_motor_MN4012KV480', 't_motor_MN4014KV330', \
                      't_motor_MN4014KV400', 't_motor_MN5208KV340', 't_motor_MN5212KV340', 't_motor_MN5212KV420', 't_motor_MT13063100KV', 't_motor_MT22081100KV', \
                      't_motor_MT2216V2800KV']}
    # do train
    do_train(constraint, disc_table)

