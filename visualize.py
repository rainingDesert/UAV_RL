import numpy as np
import matplotlib.pyplot as plt
import pdb

param_dict = {"ArmLength":(200.0, 400.0), \
            "SupportLength":(120.0,150.0), \
            "BatteryOffset_X":(-2.0, 2.0), \
            "BatteryOffset_z":(-2.0, 2.0)}

def draw_plot_custom():
    # get data
    file_name = './store/log_before445.npy'
    data_list = np.load(file_name, allow_pickle=True)
    data_list = data_list.item()
    exact_result = data_list['exact_result']
    wrong_ids = [i for i in range(len(exact_result)) if not data_list['accept'][i]]
    wrong_result = [data_list['all_result'][i] for i in wrong_ids]

    # get additional data
    file_name = './store/log.npy'
    data_list = np.load(file_name, allow_pickle=True)
    data_list = data_list.item()
    add_exact_result = data_list['exact_result']
    add_wrong_ids = [i for i in range(len(add_exact_result)) if not data_list['accept'][i]]
    add_wrong_result = [data_list['all_result'][i] for i in add_wrong_ids]

    # add
    add_wrong_ids = list(np.asarray(add_wrong_ids) + len(exact_result))
    exact_result = exact_result + add_exact_result
    wrong_ids = wrong_ids + add_wrong_ids
    wrong_result = wrong_result + add_wrong_result

    # plot
    plt.figure()
    plt.plot(np.arange(len(exact_result)), exact_result, 'k-')
    plt.plot(wrong_ids, wrong_result, 'rx')
    plt.ylim((400, 1200))
    plt.show()

def draw_param_opt():
    param_key = sorted(list(param_dict.keys()))
    # get data
    # file_name = './store/opt_log_approx_vae.npy'
    file_name = './store/opt_log_approx_ae_2.npy'
    data_list = np.load(file_name, allow_pickle=True).item()
    # param_val = data_list['real_param']
    param_val = data_list['param']
    # scale
    # process_param_val = []
    # for param in param_val:
    #     new_param = []
    #     for p_id, p in enumerate(param):
    #         # pdb.set_trace()
    #         cons = param_dict[param_key[p_id]]
    #         new_param.append(p*(cons[1]-cons[0])+cons[0])
    #     process_param_val.append(new_param)
    process_param_val = param_val
    # plot
    # plt.figure()
    fig, axs = plt.subplots(2, 2)
    for p_id in range(len(process_param_val[0])):
        cur_params = [param[p_id] for param in process_param_val]
        axs[p_id//2, p_id%2].plot(np.arange(len(cur_params)), cur_params)
    plt.show()


def draw_plot():
    # get data
    # file_name = './store/opt_log_approx.npy'
    # file_name = './store/opt_log_approx_40_20_poly_scale.npy'
    file_name = './store/opt_log_full_combine.npy'
    data_list = np.load(file_name, allow_pickle=True)
    data_list = data_list.item()
    # pdb.set_trace()
    exact_result = data_list['exact_result']
    all_result = data_list['all_result']
    # wrong_ids = [i for i in range(len(exact_result)) if not data_list['accept'][i]]
    # wrong_result = [data_list['all_result'][i] for i in wrong_ids]

    # # get add data
    # add_file_name = './store/record/opt_log_full_combine.npy'
    # add_data_list = np.load(add_file_name, allow_pickle=True).item()
    # add_exact_result = add_data_list['exact_result']
    # add_all_result = add_data_list['all_result']

    # combine
    # exact_result = list(np.asarray(add_exact_result)) + exact_result
    # all_result = list(np.asarray(add_all_result)) + all_result

    # try calculate average
    new_all_results = [np.mean(all_result[r_id:r_id+100]) for r_id in range(len(all_result))]

    # plot
    # pdb.set_trace()
    print('data size: {}'.format(len(exact_result)))
    plt.figure()
    # plt.plot(np.arange(len(exact_result)), exact_result, 'k-')
    # plt.plot(wrong_ids, wrong_result, 'rx')
    # plt.plot(np.arange(len(all_result)), all_result, 'r-')
    # plt.plot(np.arange(len(new_all_results)), new_all_results, 'k-')
    plt.plot(np.arange(len(all_result[:-100])), all_result[:-100], 'r-')
    plt.plot(np.arange(len(new_all_results[:-100])), new_all_results[:-100], 'k-')
    # plt.ylim((-100, 1e4))
    plt.show()

def draw_reject_plot():
    # get data
    file_name = './store/store_mcts/log.npy'
    data_list = np.load(file_name, allow_pickle=True).item()
    exact_result = data_list['all_result']
    # plot
    plt.figure()
    plt.plot(np.arange(len(exact_result)), exact_result)
    plt.ylim(0, 1e4)
    plt.show()

def draw_train():
    # get data
    file_name = './store/log_re.npy'
    # file_name = './store/log_ae.npy'
    data_list = np.load(file_name, allow_pickle=True)
    data_list = data_list.item()
    # pdb.set_trace()
    # plot
    column = 2
    row = (len(data_list)+1) // 2
    fig, axs = plt.subplots(row,column)
    for data_id, data_key in enumerate(data_list):
        if row == 1:
            axs[data_id%2].plot(np.arange(len(data_list[data_key])), data_list[data_key])
        else:
            axs[data_id//2, data_id%2].plot(np.arange(len(data_list[data_key])), data_list[data_key])
    plt.show()

def draw_train_vae():
    # get data
    # file_name = './store/log.npy'
    # file_name = './store/log_vae.npy'
    file_name = './store/log_full_ae.npy'
    data_list = np.load(file_name, allow_pickle=True)
    data_list = data_list.item()
    pdb.set_trace()
    # plot
    column = 2
    row = 3
    fig, axs = plt.subplots(row,column)
    for data_id, data_key in enumerate(data_list):
        loss_arr = np.asarray(data_list[data_key])
        axs[0, data_id].plot(np.arange(loss_arr.shape[0]), loss_arr[:, 0], 'k-')
        axs[1, data_id].plot(np.arange(loss_arr.shape[0]), loss_arr[:, 1], 'k-')
        axs[2, data_id].plot(np.arange(loss_arr.shape[0]), loss_arr[:, 2], 'k-')
    plt.show()

def draw_reward():
    # get data
    file_name = './store/log_re.npy'
    # file_name = './store/log_ae.npy'
    data_list = np.load(file_name, allow_pickle=True)
    data_list = data_list.item()
    pdb.set_trace()
    # get index
    train_idxs = np.argsort(data_list['train_gt'])
    valid_idxs = np.argsort(data_list['valid_gt'])
    # plot
    fig, axs = plt.subplots(2,1)
    axs[0].plot(np.arange(len(data_list['train_reward'])), np.asarray(data_list['train_reward'])[train_idxs], 'k-')
    axs[0].plot(np.arange(len(data_list['train_gt'])), np.asarray(data_list['train_gt'])[train_idxs], 'r-')
    axs[1].plot(np.arange(len(data_list['valid_reward'])), np.asarray(data_list['valid_reward'])[valid_idxs], 'k-')
    axs[1].plot(np.arange(len(data_list['valid_gt'])), np.asarray(data_list['valid_gt'])[valid_idxs], 'r-')
    plt.show()

def draw_param():
    # get data
    file_name = './store/param_record.npy'
    # file_name = './store/log_ae.npy'
    data_list = np.load(file_name, allow_pickle=True)
    data_list = data_list.item()
    # pdb.set_trace()
    # get index
    valid_pred = np.asarray(data_list['valid_param']).reshape(-1, 4)
    valid_gts = np.asarray(data_list['valid_gt']).reshape(-1, 4)
    # plot
    fig, axs = plt.subplots(4,2)
    for c_id in range(4):
        r_num = c_id // 2
        c_num = c_id % 2
        valid_pred_param = valid_pred[:, c_id]
        valid_gt_param = valid_gts[:, c_id]
        # pdb.set_trace()
        valid_idxs = np.argsort(valid_gt_param)
        axs[r_num, c_num].plot(np.arange(len(valid_idxs)), valid_pred_param[valid_idxs], 'k-')
        axs[r_num, c_num].plot(np.arange(len(valid_idxs)), valid_gt_param[valid_idxs], 'r-')\

    print(np.asarray(data_list['valid_lat']).shape)
    valid_lat = np.asarray(data_list['valid_lat']).reshape(-1, 2)
    valid_idxs = np.argsort(valid_lat[:, 0])
    axs[2,0].plot(np.arange(len(valid_idxs)), valid_lat[:, 0][valid_idxs], 'k-')
    valid_idxs = np.argsort(valid_lat[:, 1])
    axs[2,1].plot(np.arange(len(valid_idxs)), valid_lat[:, 1][valid_idxs], 'k-')
    # valid_idxs = np.argsort(valid_lat[:, 2])
    # axs[3,0].plot(np.arange(len(valid_idxs)), valid_lat[:, 2][valid_idxs], 'k-')
    # valid_idxs = np.argsort(valid_lat[:, 2])
    # axs[3,0].plot(np.arange(len(valid_idxs)), valid_lat[:, 2][valid_idxs], 'k-')
    # valid_idxs = np.argsort(valid_lat[:, 3])
    # axs[3,1].plot(np.arange(len(valid_idxs)), valid_lat[:, 3][valid_idxs], 'k-')
    plt.show()

def draw_full_param():
    # get data
    file_name = './store/full_param_record.npy'
    data_list = np.load(file_name, allow_pickle=True)
    data_list = data_list.item()
    # pdb.set_trace()
    # get index
    valid_pred = np.asarray(data_list['valid_cons_param']).reshape(-1, 4)
    valid_gts = np.asarray(data_list['valid_cons_gt']).reshape(-1, 4)
    # plot
    fig, axs = plt.subplots(5,2)
    for c_id in range(4):
        r_num = c_id // 2
        c_num = c_id % 2
        valid_pred_param = valid_pred[:, c_id]
        valid_gt_param = valid_gts[:, c_id]
        # pdb.set_trace()
        valid_idxs = np.argsort(valid_gt_param)
        axs[r_num, c_num].plot(np.arange(len(valid_idxs)), valid_pred_param[valid_idxs], 'k-')
        axs[r_num, c_num].plot(np.arange(len(valid_idxs)), valid_gt_param[valid_idxs], 'r-')\

    print(np.asarray(data_list['valid_lat']).shape)
    valid_lat = np.asarray(data_list['valid_lat']).reshape(-1, 6)
    valid_idxs = np.argsort(valid_lat[:, 0])
    axs[2,0].plot(np.arange(len(valid_idxs)), valid_lat[:, 0][valid_idxs], 'k-')
    valid_idxs = np.argsort(valid_lat[:, 1])
    axs[2,1].plot(np.arange(len(valid_idxs)), valid_lat[:, 1][valid_idxs], 'k-')
    valid_idxs = np.argsort(valid_lat[:, 2])
    axs[3,0].plot(np.arange(len(valid_idxs)), valid_lat[:, 2][valid_idxs], 'k-')
    valid_idxs = np.argsort(valid_lat[:, 3])
    axs[3,1].plot(np.arange(len(valid_idxs)), valid_lat[:, 3][valid_idxs], 'k-')
    valid_idxs = np.argsort(valid_lat[:, 4])
    axs[4,0].plot(np.arange(len(valid_idxs)), valid_lat[:, 4][valid_idxs], 'k-')
    valid_idxs = np.argsort(valid_lat[:, 5])
    axs[4,1].plot(np.arange(len(valid_idxs)), valid_lat[:, 5][valid_idxs], 'k-')
    # valid_idxs = np.argsort(valid_lat[:, 2])
    # axs[3,0].plot(np.arange(len(valid_idxs)), valid_lat[:, 2][valid_idxs], 'k-')
    # valid_idxs = np.argsort(valid_lat[:, 2])
    # axs[3,0].plot(np.arange(len(valid_idxs)), valid_lat[:, 2][valid_idxs], 'k-')
    # valid_idxs = np.argsort(valid_lat[:, 3])
    # axs[3,1].plot(np.arange(len(valid_idxs)), valid_lat[:, 3][valid_idxs], 'k-')
    plt.show()

def draw_latent_pref():
    # get data
    file_name = './store/plot_data.npy'
    data_list = np.load(file_name, allow_pickle=True)
    # get correct type
    X, Y, perfs = data_list
    pdb.set_trace()
    # draw
    plt.figure()
    ax = plt.axes(projection='3d')
    # ax.contour3D(X, Y, perfs)
    ax.plot_surface(X, Y, perfs)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

def anything():
    err_list = [0.2707499861717224, 0.004437500145286322, 0.29518750309944153, 0.0]
    plt.figure()
    plt.bar(np.arange(len(err_list)), err_list, width=0.5, color='k')
    plt.ylim((0, 0.35))
    plt.show()

if __name__ == '__main__':
    # draw_plot_custom()
    # draw_plot()
    draw_train()
    # draw_train_vae()
    # draw_reward()
    # draw_reject_plot()
    # draw_param()
    # draw_full_param()
    # draw_param_opt()
    # draw_latent_pref()
    # anything()