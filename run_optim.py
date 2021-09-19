from ae_reward.optimization import AEOpt
from ae_reward.model import AEReward
from simulator.uav_simulation import JenkinSim
from LaMCTS.LA_MCTS.lamcts import MCTS
from parsers import get_opt_parse

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
                reward_key, total_step, store_path, mode, dims, gpu, \
                url, usr_name, pwd, approx=False, vae=False):
    # load model
    if mode == 'ae-mcts':
        aer_net = AEReward(gpu, vae=vae, lat_var_dim=dims)
        if vae:
            aer_net.load_state_dict(torch.load(os.path.join(store_path, "vae.pth")))
        else:
            aer_net.load_state_dict(torch.load(os.path.join(store_path, "ae.pth")))
    elif mode == 'mcts':
        aer_net = None
    # load simulator
    if approx:
        simulator = AEReward(gpu, vae=False, lat_var_dim=dims)
        simulator.load_state_dict(torch.load(os.path.join(store_path, "after_reward.pth")))
        simulator = simulator.rnet
    else:
        # create jenkin simulation object
        simulator = JenkinSim(param_key, constraint, jenkin_param, result_path, job_name, url, usr_name, pwd)
    # get parameter constraint
    channel_fun = AEOpt(aer_net, simulator, reward_key, store_path, dims, approx)
    # test

    # optimize (MCTS)
    agent = MCTS(             
                lb = channel_fun.lbs,              # the lower bound of each problem dimensions
                ub = channel_fun.ubs,              # the upper bound of each problem dimensions
                dims = channel_fun.dims,          # the problem dimensions
                ninits = 40,      # the number of random samples used in initializations 
                func = channel_fun,               # function object to be optimized
                Cp = 40,              # Cp for MCTS
                leaf_size = 20, # tree leaf size
                kernel_type = 'sigmoid', #SVM configruation
                gamma_type = 'auto'    #SVM configruation
                # load_init = True,
                # load_path = "store/load_init_full.npy"
                )
    agent.search(total_step)


# do optimization
def do_opt(args, constraint, param_key, reward_key):
    # define parameters
    jenkin_param = {"graphGUID":"QuadCopter",\
                    "PETName":"/D_Testing/PET/FlightDyn_V1",\
                    "NumSamples":"1",\
                    "DesignVars":"\"ArmLength=200,400 SupportLength=120,150 BatteryOffset_X=-2,2 BatteryOffset_z=-1,1\""}
    job_name = "UAV_Workflows"
    # optimziation
    ae_optimize(param_key, constraint, jenkin_param, args.data_path, job_name,\
                reward_key, args.iteration, args.store_path, args.mode, args.input_dim, args.gpu, \
                args.url, args.usr_name, args.pwd, args.approx, args.vae)

# execute
if __name__ == '__main__':
    # seed
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    # common parameters
    args = get_opt_parse()
    constraint = {"ArmLength":(200.0, 400.0), \
                    "SupportLength":(120.0,150.0), \
                    "BatteryOffset_X":(-2.0, 2.0), \
                    "BatteryOffset_z":(-1.0, 1.0)}
    param_key = sorted(list(constraint.keys()))
    reward_key = 'Max_Hover_Time__s'
    # do optimization
    do_opt(args, constraint, param_key, reward_key)
