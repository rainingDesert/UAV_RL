from tqdm import tqdm
import os
import torch
import torch.nn as nn
import numpy as np
import copy
import pdb

class tracker:
    def __init__(self, foldername):
        self.all_result_list = []
        self.exact_result_list = []
        self.param_list = []
        self.real_param_list = []
        self.accept_list = []
        self.curt_best = 0

        self.foldername = foldername
        try:
            os.mkdir(foldername)
        except OSError:
            print ("Creation of the directory %s failed" % foldername)
        else:
            print ("Successfully created the directory %s " % foldername)
        
    def dump_trace(self):
        all_store = {"all_result":self.all_result_list, \
                     "exact_result":self.exact_result_list, \
                     "param":self.param_list, \
                     "real_param":self.real_param_list, \
                     "accept":self.accept_list}
        np.save(os.path.join(self.foldername,'opt_log_full.npy'), all_store, allow_pickle=True)

    def track(self, result, param, real_param=None):
        # init
        accept = False
        act_result = result
        if act_result > self.curt_best:
            accept = True
            self.curt_best = act_result
        # store
        self.all_result_list.append(act_result)
        self.exact_result_list.append(self.curt_best)
        self.accept_list.append(accept)
        self.param_list.append(list(param))
        if real_param is not None:
            self.real_param_list.append(real_param)
        # into file
        if len(self.all_result_list) % 10 == 0:
            self.dump_trace()

class AEOpt:
    def __init__(self, model, simulator, result_key, save_file, dims, approx=False, norm=True):
        self.dims = dims
        self.lbs = np.zeros(self.dims)
        self.ubs = np.ones(self.dims)
        self.result_key = result_key
        self.model = model
        self.simulator = simulator
        self.tracker = tracker(save_file)
        self.approx = approx
        self.norm = norm
    
    def __call__(self, x):
        # decode parameter
        if self.model is None:
            param = x
        else:
            with torch.no_grad():
                x = torch.from_numpy(x).float()
                param = self.model.forward_decode(x).numpy()
                print('before parameter: {}'.format(param.tolist()))
                if self.norm:
                    param = np.clip(param, 0, 1)
        # simulator
        if self.approx:
            param = torch.from_numpy(param).float().unsqueeze(0)
            result_val = self.approx_sim_result(param).item()
            print('current latent parameter: {}    |    current parameter: {}'.format(x.tolist(), param.tolist()))
        else:
            result_val = self.sim_result(param)
        # unknown reason
        if result_val >= 1e10:
            print('get unknown reason result')
            result_val = 0
        if self.model is None:
            self.tracker.track(result_val, x)
        else:
            self.tracker.track(result_val, x, param)
        # return np.log(100/(result_val+1e-9))
        if self.approx:
            return -result_val
        else:
            return -result_val/100

    # simulate to get exact reward
    def sim_result(self, param):
        # get parameters
        unnorm_param = self.simulator.unnorm_param(param)
        param_str = self.simulator.create_jenkin_param(unnorm_param)
        print('current parameter: {} \ndecoded parameter: {}'.format(param.tolist(), unnorm_param))
        # simulate to get result
        result_file = self.simulator.run_jenkins_job(param_str)
        result_val = self.simulator.get_jenkin_result(result_file, self.result_key)

        return result_val

    # get reward by approximate simulation (surrogate)
    def approx_sim_result(self, param):
        result_val = self.simulator(param)
        return result_val

class FullAEOpt:
    def __init__(self, model, auto_tool, simulator, result_key, save_file, dims, disc_table, disc_comp, comp_ori_instance, approx=False, norm=True):
        self.dims = dims
        self.lbs = np.zeros(self.dims)-1
        self.ubs = np.ones(self.dims)
        self.result_key = result_key
        self.model = model
        self.auto_tool = auto_tool
        self.simulator = simulator
        self.tracker = tracker(save_file)
        self.approx = approx
        self.norm = norm
        self.disc_table = disc_table
        self.disc_comp = disc_comp
        self.comp_ori_instance = comp_ori_instance
        # init gremlin database
        assert not approx
        print('cloneing design into CLONE_execute')
        self.auto_tool.add_clone_design(self.simulator.cur_design, "CLONE_execute")
        self.simulator.set_new_design("CLONE_execute")
    
    def __call__(self, x):
        # decode parameter
        with torch.no_grad():
            x = torch.from_numpy(x).float().unsqueeze(0)
            cons_param, disc_param = self.model.forward_decode(x)
            cons_param = cons_param.numpy().squeeze(0)
            disc_param = disc_param.numpy().squeeze(0)
            print('before cons parameter: {}  |   before disc parameter: {}'.format(cons_param.tolist(), disc_param.tolist()))
            if self.norm:
                cons_param = np.clip(cons_param, 0, 1)
        # simulator
        result_val = self.sim_result(cons_param, disc_param)
        # unknown reason
        if result_val >= 1e10:
            print('get unknown reason result')
            result_val = 0
        # store
        self.tracker.track(result_val, x, [cons_param, disc_param])
        return -result_val/100

    # simulate to get exact reward
    def sim_result(self, cons_param, disc_param):
        # swap into certain component instances
        print('swapping component...')
        for comp_id in self.disc_comp:
            comps = self.disc_comp[comp_id]
            comp_instance_id = disc_param[comp_id]
            comp_instance = self.disc_table[comp_id][comp_instance_id]
            for comp in comps:
                if type(comp) != list:
                    ori_comp = self.comp_ori_instance[comp]
                    # no need to swap (bug in autograph)
                    if ori_comp == comp_instance:
                        continue
                    self.auto_tool.add_swap_design(self.simulator.cur_design, comp, comp_instance)
                    self.comp_ori_instance[comp] = comp_instance
                # current specific for "propeller"
                else:
                    if "Prop_0" in comp:
                        ori_comp = self.comp_ori_instance["Prop_0"]
                        # no need to swap (bug in autograph)
                        if ori_comp == comp_instance:
                            continue
                        self.auto_tool.add_swap_design(self.simulator.cur_design, "Prop_0", comp_instance)
                        self.auto_tool.add_swap_design(self.simulator.cur_design, "Prop_2", comp_instance)
                        self.comp_ori_instance["Prop_0"] = comp_instance
                        self.comp_ori_instance["Prop_2"] = comp_instance
                    elif "Prop_1" in comp:
                        ori_comp = self.comp_ori_instance["Prop_1"]
                        # no need to swap (bug in autograph)
                        if ori_comp == comp_instance[:-1]:
                            continue
                        self.auto_tool.add_swap_design(self.simulator.cur_design, "Prop_1", comp_instance[:-1])
                        self.auto_tool.add_swap_design(self.simulator.cur_design, "Prop_3", comp_instance[:-1])
                        self.comp_ori_instance["Prop_1"] = comp_instance[:-1]
                        self.comp_ori_instance["Prop_3"] = comp_instance[:-1]
                    else:
                        print("?????")
                        pdb.set_trace()
                        pass
        self.auto_tool.execute()
        # get parameters
        print('simulate...')
        unnorm_param = self.simulator.unnorm_param(cons_param)
        param_str = self.simulator.create_jenkin_param(unnorm_param)
        print('current parameter: {} \ndecoded parameter: {}'.format(cons_param.tolist(), unnorm_param))
        # simulate to get result
        result_file = self.simulator.run_jenkins_job(param_str)
        result_val = self.simulator.get_jenkin_result(result_file, self.result_key)

        return result_val

class FullAEOptRun:
    def __init__(self, auto_tool, simulator, result_key, disc_table, disc_comp, comp_ori_instance, parallel_num):
        self.result_key = result_key
        self.auto_tool = auto_tool
        self.simulator = simulator
        self.disc_table = disc_table
        self.disc_comp = disc_comp
        self.comp_ori_instance = comp_ori_instance
        self.comp_ori_instance_dict = {}
        self.parallel_num = parallel_num
        # init gremlin database
        print('cloneing design into CLONE_execute')
        for p_id in range(self.parallel_num):
            self.auto_tool.add_clone_design(self.simulator.cur_design, "CLONE_execute_{}".format(p_id))
            # self.simulator.set_new_design("CLONE_execute")
            self.comp_ori_instance_dict["CLONE_execute_{}".format(p_id)] = copy.deepcopy(self.comp_ori_instance)
    
    # simulate to get exact reward
    def sim_result(self, cons_params, disc_params, run_parallel):
        assert run_parallel <= self.parallel_num
        # swap into certain component instances
        print('swapping component...')
        for p_id in range(run_parallel):
            # get related item
            design_key = "CLONE_execute_{}".format(p_id)
            comp_ori_instance = self.comp_ori_instance_dict[design_key]
            # do edit
            for comp_id in self.disc_comp:
                comps = self.disc_comp[comp_id]
                comp_instance_id = disc_params[p_id][comp_id]
                comp_instance = self.disc_table[comp_id][comp_instance_id]
                for comp in comps:
                    if type(comp) != list:
                        ori_comp = comp_ori_instance[comp]
                        # no need to swap (bug in autograph)
                        if ori_comp == comp_instance:
                            continue
                        self.auto_tool.add_swap_design(design_key, comp, comp_instance)
                        comp_ori_instance[comp] = comp_instance
                    # current specific for "propeller"
                    else:
                        if "Prop_0" in comp:
                            ori_comp = comp_ori_instance["Prop_0"]
                            # no need to swap (bug in autograph)
                            if ori_comp == comp_instance:
                                continue
                            self.auto_tool.add_swap_design(design_key, "Prop_0", comp_instance)
                            self.auto_tool.add_swap_design(design_key, "Prop_2", comp_instance)
                            comp_ori_instance["Prop_0"] = comp_instance
                            comp_ori_instance["Prop_2"] = comp_instance
                        elif "Prop_1" in comp:
                            ori_comp = comp_ori_instance["Prop_1"]
                            # no need to swap (bug in autograph)
                            if ori_comp == comp_instance[:-1]:
                                continue
                            self.auto_tool.add_swap_design(design_key, "Prop_1", comp_instance[:-1])
                            self.auto_tool.add_swap_design(design_key, "Prop_3", comp_instance[:-1])
                            comp_ori_instance["Prop_1"] = comp_instance[:-1]
                            comp_ori_instance["Prop_3"] = comp_instance[:-1]
                        else:
                            print("?????")
                            pdb.set_trace()
                            pass
        self.auto_tool.execute()
        # get parameters
        print('simulate...')
        param_list = []
        for p_id in range(run_parallel):
            unnorm_param = self.simulator.unnorm_param(cons_params[p_id])
            self.simulator.set_new_design("CLONE_execute_{}".format(p_id))
            param_str = self.simulator.create_jenkin_param(unnorm_param)
            print('parallel id {} : current parameter: {} \ndecoded parameter: {}'.format(p_id, cons_params[p_id].tolist(), unnorm_param))
            param_list.append(param_str)
        # simulate to get result
        result_files = self.simulator.run_jenkins_job_parallel(param_list)
        if run_parallel == 1:
            result_files = result_files[0]
        result_val = self.simulator.get_jenkin_result(result_files, self.result_key, file_num=run_parallel)

        return result_val