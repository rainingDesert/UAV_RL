import jenkins
from  jenkinsapi.jenkins import Jenkins as Jenkin_tool
from autograph.AutoGraph_Call import run_autograph
from gremlin_python.driver import client
# import jenkinsapi.jenkins.Jenkins as Jenkin_tool
import numpy as np
import pandas as pd
import os
import copy
import time
import pandas as pd

import pdb

class JenkinSim:
    def __init__(self, url, usr_name, password, param_key, constraint, jenkin_param, result_path, job_name, normalize_request=True):
        self.normalize_request = normalize_request
        self.sleep_max_step = 10
        self.jenkin_param = jenkin_param
        self.url = url
        self.usr_name = usr_name
        self.password = password
        self.j_obj = self.define_jenkins()
        self.constraint = constraint
        self.param_key = param_key
        self.result_path = result_path
        self.job_name = job_name
        # new define
        self.new_designs = []
        self.cur_design = jenkin_param['graphGUID']
        self.default_param = self.get_all_param()

    # hard write for now
    def define_jenkins(self):
        url = self.url
        usr_name = self.usr_name
        jenkin_obj = jenkins.Jenkins(url, username=self.usr_name, password=self.password)
        return jenkin_obj

    # get all defined parameters under jenkin_param
    def get_all_param(self):
        all_design = self.jenkin_param['DesignVars'][1:-1].split(' ')
        param_dict = {param.split('=')[0]:param.split('=')[1] for param in all_design}
        return param_dict


    # build jobs in parallel and return after finish
    def build_execute_parallel(self, job_names, param_lists, sleep_time):
        # init
        cur_step = 0
        fail_build = []
        # each job
        for cur_job, param in zip(job_names, param_lists):
            self.j_obj.build_job(cur_job, param)
        # wait execute
        while True:
            # wait
            time.sleep(sleep_time)
            cur_step += 1
            # check
            build_queue = self.j_obj.get_running_builds()
            if len(build_queue) == 0:
                break
            # end
            if cur_step >= self.sleep_max_step:
                # stop unfinished jobs
                build_queue = self.j_obj.get_running_builds()
                for job in build_queue:
                    fail_build.append(job['name'])
                    self.j_obj.stop_build(job['name'], job['number'])
                break
        return fail_build

    # run jenkins in parallel
    def run_jenkins_job_parallel(self, param_lists, sleep_time=150):
        # param
        parallel_time = len(param_lists)
        assert len(self.job_name) >= parallel_time
        job_names = self.job_name[:parallel_time]
        result_path = self.result_path
        before_file_num = [len(os.listdir(path)) for path in result_path[:parallel_time]]
        # get final result
        filename_list = []
        fail_build = self.build_execute_parallel(job_names, param_lists, sleep_time)
        # get results
        for p_id in range(parallel_time):
            # if fail
            if job_names[p_id] in fail_build:
                filename_list.append(None)
                print("None for {} as job {}".format(p_id, job_names[p_id]))
                continue
            # also fail
            cur_list = os.listdir(result_path[p_id])
            if len(cur_list) == before_file_num[p_id]:
                filename_list.append(None)
                print("None for {} as job {}".format(p_id, job_names[p_id]))
                continue
            # else, find file name
            cur_list = sorted(cur_list, reverse=True)
            for file_name in cur_list:
                if file_name[:5] == 'r2021':
                    print(os.path.join(result_path[p_id], file_name, 'output.csv'))
                    filename_list.append(file_name)
                    break
        return filename_list


    def run_jenkins_job(self, params, sleep_time=120):
        filename_list = self.run_jenkins_job_parallel([params], sleep_time)
        return filename_list[0]

    # create parameter for jenkin based on value
    def create_jenkin_param(self, param_list):
        # param
        param_key = self.param_key
        param_str = copy.deepcopy(self.jenkin_param)
        new_design_vars = "\""
        for param_val, param_name in zip(param_list, param_key):
            low_val = param_val
            up_val = param_val
            new_design_vars = new_design_vars + "{}={},{} ".format(param_name, low_val, up_val)
        # load default param
        for param_name in self.default_param:
            if param_name not in param_key:
                new_design_vars = new_design_vars + "{}={} ".format(param_name, self.default_param[param_name])
        # end char
        new_design_vars = new_design_vars[:-1] + "\""
        param_str["DesignVars"] = new_design_vars
        return param_str

    # normalize parameter values to [0, 1]
    def norm_param(self, param_list):
        # init
        param_key = self.param_key
        new_param_list = []
        # norm
        for param_val, cur_key in zip(param_list, param_key):
            norm_param_val = param_val - self.constraint[cur_key][0]
            norm_param_val = norm_param_val / (self.constraint[cur_key][1] - self.constraint[cur_key][0])
            new_param_list.append(norm_param_val)
        return new_param_list

    # map back from normalized parameter values
    def unnorm_param(self, param_list):
        # init
        param_key = self.param_key
        new_param_list = []
        # norm
        for param_val, cur_key in zip(param_list, param_key):
            unnorm_param_val = param_val * (self.constraint[cur_key][1] - self.constraint[cur_key][0])
            unnorm_param_val = unnorm_param_val + self.constraint[cur_key][0]
            new_param_list.append(unnorm_param_val)
        return new_param_list

    def get_jenkin_result(self, file_name, key_name, file_num=1):
        if file_num == 1:
            if file_name is None:
                return 0.0
            else:
                data_path = os.path.join(self.result_path[0], file_name, "output.csv")
                df = pd.read_csv(data_path, header=0)
                return df[key_name].values.item()
        else:
            reward_list = []
            for f_id, f in enumerate(file_name):
                if f is None:
                    reward_list.append(0.0)
                else:
                    data_path = os.path.join(self.result_path[f_id], f, "output.csv")
                    df = pd.read_csv(data_path, header=0)
                    reward_list.append(df[key_name].values.item())
            return reward_list

    def get_param_bounds(self):
        if self.normalize_request:
            lbs = list(np.zeros(len(self.param_key)))
            ubs = list(np.ones(len(self.param_key)))
        else:
            lbs = [self.constraint[param][0] for param in self.param_key]
            ubs = [self.constraint[param][1] for param in self.param_key]
        return lbs, ubs

    def set_new_design(self, design_name):
        self.jenkin_param['graphGUID'] = design_name
        self.cur_design = design_name


class AutoTool:
    def __init__(self, temp_file_name, hostIP='localhost'):
        self.temp_file_name = temp_file_name
        self.doc_temp = {'Qtemplate':[], 'Param1Name':[], 'Param1Val':[],\
                         'Param2Name':[], 'Param2Val':[], 'Param3Name':[], 'Param3Val':[],\
                         'Param4Name':[], 'Param4Val':[], 'Param5Name':[], 'Param5Val':[]}
        self.new_doc_temp = copy.deepcopy(self.doc_temp)
        self.client = client.Client('ws://'+hostIP+':8182/gremlin', 'g')
        self.cur_len = 0
        
    def new_line(self):
        for doc_key in self.new_doc_temp:
            self.new_doc_temp[doc_key].append(None)
        self.cur_len += 1

    def add_clone_design(self, ori_design, after_design):
        self.new_line()
        self.new_doc_temp['Qtemplate'][self.cur_len-1] = 'CloneDes'
        self.new_doc_temp['Param1Name'][self.cur_len-1] = "__SOURCENAME__"
        self.new_doc_temp['Param1Val'][self.cur_len-1] = ori_design
        self.new_doc_temp['Param2Name'][self.cur_len-1] = "__DESTNAME__"
        self.new_doc_temp['Param2Val'][self.cur_len-1] = after_design

    def add_swap_design(self, design_name, comp_inst, after_comp):
        self.new_line()
        self.new_doc_temp['Qtemplate'][self.cur_len-1] = 'swap'
        # self.new_doc_temp['Param1Name'][self.cur_len-1] = "__SOURCEDESIGN__"
        self.new_doc_temp['Param1Name'][self.cur_len-1] = "__DESIGN__"
        self.new_doc_temp['Param1Val'][self.cur_len-1] = design_name
        # self.new_doc_temp['Param2Name'][self.cur_len-1] = "__SOURCECOMP__"
        self.new_doc_temp['Param2Name'][self.cur_len-1] = "__COMPONENT_INSTANCE__"
        self.new_doc_temp['Param2Val'][self.cur_len-1] = comp_inst
        self.new_doc_temp['Param3Name'][self.cur_len-1] = "__NEW_COMPONENT__"
        self.new_doc_temp['Param3Val'][self.cur_len-1] = after_comp
    
    def clear_design(self, design_name):
        self.new_line()
        self.new_doc_temp['Param1Name'][self.cur_len-1] = "__SOURCEDESIGN__"
        self.new_doc_temp['Param1Val'][self.cur_len-1] = design_name

    def execute(self):
        pd_data = pd.DataFrame(self.new_doc_temp)
        pd_data.to_csv(self.temp_file_name, index=None)
        run_autograph(client=self.client, fileName = self.temp_file_name)
        self.new_doc_temp = copy.deepcopy(self.doc_temp)
        self.cur_len = 0