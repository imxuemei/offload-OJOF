# coding=utf-8
import argparse
import time
import os
import OJOF_ucache_v5.env_constants_v2 as mycst
from OJOF_ucache_v5 import evaluation


def run(args):
    N, M, T, cache_ratio = args.N, args.M, args.T, args.cr
    log_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    zipf_params = args.zipf if args.zipf is not None else [0.56, 1.0]
    seed = args.seed
    # np.random.seed(seed)

    fw_methods = ["GP", "OJOF", "LP", "LN", "EP", "EN", "RP", "RN"]
    o_methods = [mycst.OFFLOAD_GREEDY, mycst.MAB_I_LINUCB, mycst.MAB_LINUCB, mycst.MAB_E_GREEDY, mycst.MAB_UCB1]
    # o_methods = [mycst.MAB_I_LINUCB]
    c_methods = [mycst.CA_PERFECT, mycst.CA_TLCPD, mycst.CA_W_LFU, mycst.CA_LFU]

    if args.eval == 'framework':
        print("====== start eval different frameworks ======")
        fw_data_file = f"{args.data_path}/fw-{T}-{N}-{M}.npz"
        if not os.path.exists(fw_data_file):
            fw_data_file = None
        evaluation.test_framework_methods(fw_methods, T, N, M, cache_ratio, zipf_params, data_file=fw_data_file,
                                          log_dir=f"{args.log}/{log_time}-fw-{T}-{N}-{M}-{cache_ratio}-{seed}/",
                                          seed=seed)
        print("====== end eval different frameworks ======")

    elif args.eval == 'fw-diff-devs':
        # the framework performance under different devs
        print("====== start eval fw-diff-devs ======")
        N_list = [10, 20, 30, 40, 50] if args.N_list is None else args.N_list
        # N_list = [10, 20]
        fw_diffd_files = [f"{args.data_path}/fw-{T}-{n}-{M}.npz" for n in
                          N_list] if args.data_path is not None else None
        evaluation.test_fw_diff_devs(fw_methods, T, N_list, M, cache_ratio, zipf_params, data_files=fw_diffd_files,
                                     log_dir=f"{args.log}/{log_time}-fw-diff-devs-{T}-{M}-{cache_ratio}-{seed}/",
                                     seed=seed)
        print("====== end eval fw-diff-devs ======")

    elif args.eval == 'fw-diff-tasks':
        # the framework performance under different tasks
        print("====== start eval fw-diff-tasks ======")
        M_list = [10, 30, 50, 70, 90] if args.M_list is None else args.M_list

        fw_difft_files = [f"{args.data_path}/fw-{T}-{N}-{m}.npz" for m in
                          M_list] if args.data_path is not None else None
        evaluation.test_fw_diff_tasks(fw_methods, T, N, M_list, cache_ratio, zipf_params, data_files=fw_difft_files,
                                      log_dir=f"{args.log}/{log_time}-fw-diff-tasks-{T}-{N}-{cache_ratio}-{seed}/",
                                      seed=seed)
        print("====== end eval fw-diff-tasks ======")

    elif args.eval == 'offload':
        print("====== start eval different offload methods ======")
        o_data_file = f"{args.data_path}/offload-{T}-{N}-{M}.npz"
        if not os.path.exists(o_data_file):
            o_data_file = None
        evaluation.test_offload_methods(o_methods, T, N, M, cache_ratio, zipf_params, data_file=o_data_file,
                                        log_dir=f"{args.log}/{log_time}-o-{T}-{N}-{M}-{cache_ratio}-{seed}/", seed=seed)
        print("====== end eval different offload methods ======")

    elif args.eval == 'o-diff-crs':
        print("====== start eval o-diff-caches ======")
        cr_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] if args.cr_list is None else args.cr_list

        o_data_file = f"{args.data_path}/offload-{T}-{N}-{M}.npz"
        if not os.path.exists(o_data_file):
            o_data_file = None
        evaluation.test_o_diff_caches(o_methods, T, N, M, cr_list, zipf_params, data_file=o_data_file,
                                      log_dir=f"{args.log}/{log_time}-o-diff-crs-{T}-{N}-{M}-{seed}/", seed=seed)
        print("====== end eval o-diff-caches ======")

    elif args.eval == 'cache':
        print("====== start eval different cache methods ======")
        if args.zipf is None:
            zipf_params = [0.56, 0.7, 1.0, 0.56, 0.7]
            T = 5000 * len(zipf_params)

        c_data_file = f"{args.data_path}/cache-{T}-{N}-{M}-ROSS.npz"
        if not os.path.exists(c_data_file):
            c_data_file = None

        evaluation.test_cache_methods(c_methods, T, N, M, cache_ratio, zipf_params, data_file=c_data_file,
                                      log_dir=f"{args.log}/{log_time}-c-{T}-{N}-{M}-{cache_ratio}-{seed}/", seed=seed,
                                      figsize=(5, 3.0))
        print("====== end eval different cache methods ======")

    elif args.eval == 'c-diff-patterns':
        print("====== start eval different cache methods in different patterns ======")
        pop_patterns = ["SOSS", "ROSS", "RORS", "RORSRP"]
        for pattern in pop_patterns:
            c_data_file = f"{args.data_path}/cache-{T}-{N}-{M}-{pattern}.npz" if args.data_path is not None else None
            evaluation.test_cache_methods(c_methods, T, N, M, cache_ratio, zipf_params, c_data_file,
                                          log_dir=f"{args.log}/{log_time}-c-diff-patterns-{T}-{N}-{M}-{cache_ratio}-{pattern}-{seed}/", seed=seed,
                                          pattern=pattern)
        print("====== end eval different cache methods in different patterns ======")
    else:
        raise ValueError(
            "only support framework, fw-diff-devs, fw-diff-tasks, offload, o-diff-crs, cache, c-diff-patterns")


parser = argparse.ArgumentParser()
parser.add_argument('--eval', type=str, default='fw-diff-devs',
                    help='values can be framework, fw-diff-devs, fw-diff-tasks, offload, o-diff-caches, cache, c-diff-patterns')
parser.add_argument('--N', type=int, default=10, help='the number of devices')
parser.add_argument('--M', type=int, default=50, help='the number of tasks')
parser.add_argument('--T', type=int, default=10000, help='the number of time slots')
parser.add_argument('--cr', type=float, default=0.4, help='the cache ratio of the task_config')
parser.add_argument('--zipf', nargs='+', type=float, help='different zipf parameters on each change point')
parser.add_argument('--log', type=str, default='./output/', help='the log files path')
parser.add_argument('--data_path', type=str, default='./data/', help='the data files path')
parser.add_argument('--N_list', nargs='+', type=int, help='different number of devices')
parser.add_argument('--M_list', nargs='+', type=int, help='different number of task_config')
parser.add_argument('--cr_list', nargs='+', type=float, help='different cache ratios')
parser.add_argument('--seed', type=int, default=200)

if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
