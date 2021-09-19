import numpy as np
import pandas as pd
import pdb

class DataCheck:
    # init
    def __init__(self, data_file):
        self.hist_data = pd.read_csv(data_file, header=0)

    # get length
    def __len__(self):
        return self.hist_data.shape[0]

    # check invalid
    def check(self, data_dict):
        check_list = self.hist_data
        for data_key in data_dict:
            check_list = check_list[check_list[str(data_key)]==data_dict[data_key]]
            # not in list
            if check_list.shape[0] == 0:
                return 2
        # in list
        if check_list['valid'].item() == 1:
            return 1
        else:
            return 0

    # add new data
    def add_data(self, new_data, valid):
        if self.check(new_data) != 2:
            return
        new_data['valid'] = valid
        self.hist_data = self.hist_data.append(new_data, ignore_index=True)

    # save data
    def store_data(self, data_path):
        self.hist_data.to_csv(data_path, index=False)