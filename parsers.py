import argparse

def get_train_parse():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='use gpu or not')
    # train
    parser.add_argument('--reward_lr', type=float, default=0.1, 
                        help='learning rate to train reward network')
    parser.add_argument('--reward_momentum', type=float, default=0.95, 
                        help='momentum to train reward network')
    parser.add_argument('--model_lr', type=float, default=0.1, 
                        help='learning rate to train entire model')
    parser.add_argument('--model_momentum', type=float, default=0.9, 
                        help='momentum to train entire model')
    parser.add_argument('--log_path', type=str, required=False,
                        help='path to save log')
    # data
    parser.add_argument('--csv_file', type=str, required=False, default=None,
                        help='csv file to load data')
    parser.add_argument('--train_ratio', type=str, default=0.8,
                        help='ratio of train data vs. validation data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size of train set')
    parser.add_argument('--num_worker', type=int, default=0,
                        help='number of workers for datalaoder')
    parser.add_argument('--re_ite', type=int, default=10,
                        help='train epochs for reward network')
    parser.add_argument('--ae_ite', type=int, default=10,
                        help='train epochsa for entire model')
    # model
    parser.add_argument('--reward_path', type=str, required=False, default=None,
                        help='path to store model after reward training')
    parser.add_argument('--model_path', type=str, required=False, default=None,
                        help='path to store model after entire training')
    parser.add_argument('--vae', action='store_true', default=False,
                        help='use vae instead of ae')
    
    return parser.parse_args()

def get_rl_parse():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='use gpu or not')
    # train
    parser.add_argument('--load_path', type=str, default=None,
                        help='if specific, load model before train')
    parser.add_argument('--parallel_num', type=int, default=1,
                        help='number of parallel')
    parser.add_argument('--iteration', type=int, default=1000,
                        help='iterations to train controller')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate to train controller')
    parser.add_argument('--momentum', type=float, default=0.9, 
                        help='momentum to train entire model')
    parser.add_argument('--log_path', type=str, required=False,
                        help='path to save log')
    # data
    parser.add_argument('--data_path', type=str, default="D:\\jwork\\Agents\\workspace\\UAV_Workflows\\results",
                        help='path to load simulation data')
    parser.add_argument('--csv_file', type=str, default=None,
                        help='previous data to check valid')
    parser.add_argument('--new_csv_file', type=str, default=None,
                        help='data file to store updated data')
    parser.add_argument('--sample_num', type=int, default=1,
                        help='sample number for architecture')
    # model
    parser.add_argument('--model_path', type=str, default=None,
                        help='path to store controller')
    # simulator
    parser.add_argument('--url', type=str, default='http://localhost:8080',
                        help='url of jenkin')
    parser.add_argument('--usr_name', type=str, default='gc669',
                        help='user name for jenkin')
    parser.add_argument('--password', type=str, default='gc669',
                        help='password for jenkin')

    return parser.parse_args()


def get_opt_parse():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='use gpu or not')
    # optimize
    parser.add_argument('--iteration', type=int, default=300,
                        help='total iteration for optimization')
    parser.add_argument('--data_path', type=str, default="D:\\jwork\\Agents\\workspace\\UAV_Workflows\\results",
                        help='path to load simulation data')
    parser.add_argument('--mode', type=str, required=True, choices=["mcts", "ae-mcts"],
                        help='choose a specific optimization mode')   
    parser.add_argument('--approx', action='store_true', default=False,
                        help='use approximate method to predict reward if set') 
    # model
    parser.add_argument('--store_path', type=str, required=True, default=None,
                        help='path to store models and logs')
    parser.add_argument('--vae', action='store_true', default=False,
                        help='use vae instead of ae')
    
    return parser.parse_args()