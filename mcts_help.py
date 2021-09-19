import numpy as np
import pdb

def load_init_datas_uav():
    # previous searched
    prev_all_results = np.load("store/opt_log_before80.npy", allow_pickle=True).item()
    # new searched
    new_all_results = np.load("store/opt_log.npy", allow_pickle=True).item()
    new_param_points = prev_all_results['param'] + new_all_results['param']
    new_results = prev_all_results['all_result'] + new_all_results['all_result']
    # new_load_points = [(np.asarray(p), np.log(1000/(r+1e-9))) for p, r in zip(new_param_points, new_results)]
    new_load_points = [(np.asarray(p), -r/100) for p, r in zip(new_param_points, new_results)]
    # combine
    load_points = new_load_points
    np.save("store/load_init.npy", load_points)

def load_init_datas_full_uav():
    # searched
    all_results = np.load("store/record/opt_log_full_combine.npy", allow_pickle=True).item()
    param_points = all_results['param']
    results = all_results['all_result']
    load_points = [(p[0].numpy(), -r/100) for p, r in zip(param_points, results)]
    np.save("store/load_init_full.npy", load_points)

def combine_records():
    # load
    all_results_1 = np.load("store/record/opt_log_full_combine.npy", allow_pickle=True).item()
    all_results_2 = np.load("store/opt_log_full.npy", allow_pickle=True).item()
    all_results = {}
    # combine
    for key in all_results_1:
        all_results[key] = all_results_1[key] + all_results_2[key]
    np.save("store/opt_log_full_combine.npy", all_results)


if __name__ == "__main__":
    # load_init_datas_uav()
    # load_init_datas_full_uav()
    combine_records()