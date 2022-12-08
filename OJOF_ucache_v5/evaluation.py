# coding=utf-8
import os.path

import numpy as np
import OJOF_ucache_v5.env_constants_v2 as mycst
import OJOF_ucache_v5.eps_data as myeps
from OJOF_ucache_v5.env_config_v2 import EdgeConfig, DeviceConfig, TaskConfig, load_data, save_data
from OJOF_ucache_v5.device_agent_v5 import DeviceAgentV5
from OJOF_ucache_v5.edge_agent_v5 import EdgeAgentV5, EdgeAgentPerfectV5, EdgeAgentNoCacheV5


def generate_zipf_probs(num_ranks, gamma):
    probs = np.zeros(num_ranks)
    for i in range(num_ranks):
        probs[i] = pow(1 / (i + 1), gamma)
    probs = probs / np.sum(probs)
    return probs


def generate_random_probs(num_ranks):
    pops = np.random.randint(10, num_ranks * 10, size=num_ranks)
    return pops / np.sum(pops)


def generate_req_tids_seq(T, N, M, zipf_params=None, pattern="ROSS"):
    if pattern == "SOSS":
        random_task_order, random_change_point = False, False
    elif pattern == "ROSS":
        random_task_order, random_change_point = True, False
    elif pattern == "RORS":
        random_task_order, random_change_point = True, True
    elif pattern == "RORSRP":
        random_task_order, random_change_point = True, True
        zipf_params = None

    num_changes = 2 if zipf_params is None else len(zipf_params)
    # gen the task pops of each user, cp_pops=(N, num_changes, M)
    cp_pops = []
    for n in range(N):
        n_pops = []
        for i in range(num_changes):
            if zipf_params is None:
                i_pops = generate_random_probs(M)
            else:
                i_pops = generate_zipf_probs(M, zipf_params[i])
            if random_task_order:
                np.random.shuffle(i_pops)
            n_pops.append(i_pops)
        cp_pops.append(n_pops)
    cp_pops = np.reshape(cp_pops, (N, num_changes, M))

    # gen the change point step of each user, cp_steps=(N, num_changes)
    cp_span, user_span = int(T / num_changes), int(T / num_changes / N)
    if random_change_point:
        cp_steps = np.zeros((N, num_changes), dtype=np.int32)
        for n in range(N):
            for i in range(num_changes):
                i_step = int((i + 0.5) * cp_span)
                s_step = i_step + n * user_span
                e_step = s_step + user_span
                cp_steps[n, i] = np.random.randint(s_step, e_step)
        cp_steps[:, -1] = T
    else:
        n_cp_steps = np.cumsum([cp_span for _ in range(num_changes)])
        n_cp_steps[-1] = T
        cp_steps = np.repeat([n_cp_steps], repeats=N, axis=0)

    # gen req_tids_seq
    req_tids_seq = []
    for n in range(N):
        n_pops = cp_pops[n]
        n_steps = cp_steps[n]
        n_req_tids = np.array([], dtype=np.int32)
        for i in range(num_changes):
            ni_size = n_steps[i] if i == 0 else (n_steps[i] - n_steps[i - 1])
            ordered_tids = np.arange(M, dtype=np.int32)
            ni_req_tids = np.random.choice(ordered_tids, p=n_pops[i], size=ni_size)  # 按probs采样
            n_req_tids = np.concatenate((n_req_tids, ni_req_tids))
        req_tids_seq.append(n_req_tids)
    req_tids_seq = np.transpose(np.reshape(req_tids_seq, (N, T)))
    cp_config = {
        'cp_pops': cp_pops,
        'cp_steps': cp_steps,
    }
    return req_tids_seq, cp_config


def offload_and_cache(req_tids_seq: np.ndarray, edge_agent: EdgeAgentV5, dev_agent_list: list):
    T, N = req_tids_seq.shape[0], req_tids_seq.shape[1]
    avg_edge_cpu = edge_agent.allocate_cpu(N)
    c2e_rate = edge_agent.get_c2e_rate()

    for t in range(T):
        cur_req_tids = req_tids_seq[t]
        cur_cache_list = edge_agent.get_cache_list()
        edge_agent.update_hit_stat(cur_req_tids)
        edge_agent.receive_tasks(cur_req_tids)

        # 1 select arms of the offload methods of each device
        arm_dict_list = []
        local_delays, avg_remote_delays = np.zeros(N), np.zeros(N)
        for n in range(N):
            tid = cur_req_tids[n]
            cache_status = mycst.CACHED if tid in cur_cache_list else mycst.NO_CACHE

            arm_dict, local_delays[n], avg_remote_delays[n] = dev_agent_list[n].select_arms(t, tid, cache_status,
                                                                                            c2e_rate, avg_edge_cpu)
            arm_dict_list.append(arm_dict)

        # 2 compute actual remote delays for bandit offload methods
        rN_dict = {}
        for bandit_type in dev_agent_list[0].get_bandit_types():
            if bandit_type == mycst.OFFLOAD_LOCAL:
                rN_dict[bandit_type] = 0
            elif bandit_type == mycst.OFFLOAD_REMOTE:
                rN_dict[bandit_type] = N
            else:
                r_Ns = [1 if arm_dict_list[n][bandit_type] in [mycst.ARM_C_R, mycst.ARM_NOC_R] else 0 for n in range(N)]
                rN_dict[bandit_type] = np.sum(r_Ns)

        delay_dict_list = [{} for _ in range(N)]
        for bandit_type in dev_agent_list[0].get_bandit_types():
            if rN_dict[bandit_type] == 0:
                for n in range(N):
                    delay_dict_list[n][bandit_type] = local_delays[n]
            elif rN_dict[bandit_type] == N:
                for n in range(N):
                    delay_dict_list[n][bandit_type] = avg_remote_delays[n]
            else:
                shared_edge_cpu = edge_agent.allocate_cpu(rN_dict[bandit_type])
                for n in range(N):
                    if arm_dict_list[n][bandit_type] in [mycst.ARM_C_R, mycst.ARM_NOC_R]:
                        # compute new remote delays
                        tid = cur_req_tids[n]
                        cache_status = mycst.CACHED if tid in cur_cache_list else mycst.NO_CACHE
                        delay_dict_list[n][bandit_type] = dev_agent_list[n].compute_remote_delay(t, tid,
                                                                                                 cache_status,
                                                                                                 c2e_rate,
                                                                                                 shared_edge_cpu)
                    else:
                        delay_dict_list[n][bandit_type] = local_delays[n]

        # 3 pull arms
        for n in range(N):
            dev_agent_list[n].pull_arms(cur_req_tids[n], arm_dict_list[n], delay_dict_list[n])
    # the offloading and caching is finished here


def get_offload_metric(dev_agent_list: list):
    # 需要每个device的bandit类型一致
    N = len(dev_agent_list)
    bandit_types = dev_agent_list[0].get_bandit_types()
    # 获得所有设备metric
    cum_d_hists, cum_r_hists = [], []
    for n in range(N):
        cum_delay_hists, cum_regret_hists, suc_ratio_hists = dev_agent_list[n].get_metrics()
        cum_d_hists.append(cum_delay_hists)
        if cum_regret_hists is not None:
            cum_r_hists.append(cum_regret_hists)
    # 求和所有设备metric
    # cum_d_hist_dict = {}
    slot_cum_d_hist_dict = {}
    for bt in bandit_types:
        cum_d = 0
        for n in range(N):
            cum_d = cum_d + np.array(cum_d_hists[n][bt])
        slot_cum_d = cum_d / np.arange(1, 1 + len(cum_d))
        slot_cum_d_hist_dict[bt] = slot_cum_d

    cum_r_hist_dict = None
    if len(cum_r_hists) > 0:
        cum_r_hist_dict = {}
        for bt in bandit_types:
            cum_r = 0
            for n in range(N):
                cum_r = cum_r + np.array(cum_r_hists[n][bt])
            cum_r_hist_dict[bt] = cum_r

    return slot_cum_d_hist_dict, cum_r_hist_dict


def get_cache_metric(edge_agent_dict: dict):
    # 统计不同设置下edge agent的缓存size，以及命中率
    cum_hs_hist_dict, slot_cum_hs_hist_dict = {}, {}
    for ename, eagent in edge_agent_dict.items():
        cum_hit_size_hist, cum_slot_hs_hist, cum_hit_num_hist, cum_slot_hn_hist = eagent.get_metrics()
        cum_hs_hist_dict[ename] = cum_hit_size_hist
        slot_cum_hs_hist_dict[ename] = cum_slot_hs_hist
    return cum_hs_hist_dict, slot_cum_hs_hist_dict


#############################################################################
# all the tests
#############################################################################

def test_framework_methods(fw_methods, T, N, M, cache_ratio, zipf_params, data_file=None, log_dir="./logs/", seed=None):
    """
    metrics: fw_scd_hists, fw_chs_hists, fw_schs_hists.
    :param fw_methods = ["LP", "LN", "EP", "EN", "RP", "RN", "PP", "OJOF"]
    :param req_tids_seq: (T, N)
    :param task_config:
    :param req_pops: (N, num_changes, M)
    :param log_dir:
    :return:
    """
    if data_file is None:
        np.random.seed(seed)
        task_config = TaskConfig(M, cache_ratio)
        dev_configs = [DeviceConfig(T) for _ in range(N)]
        req_tids_seq, cp_config = generate_req_tids_seq(T, N, M, zipf_params, pattern="ROSS")
    else:
        dev_configs, task_config, req_tids_seq, cp_config = load_data(data_file, cache_ratio)
    edge_config = EdgeConfig(task_config.max_cache_size)

    fw_scd_hists, fw_chs_hists, fw_schs_hists = [], [], []
    for method in fw_methods:
        if method == "OJOF":
            edge_agent = EdgeAgentV5(edge_config, task_config, mycst.DEFAULT_CA_PARAMS[mycst.CA_TLCPD])
        elif method in ["LP", "EP", "RP", "GP"]:
            edge_agent = EdgeAgentPerfectV5(edge_config, task_config, cp_config)
        elif method in ["LN", "EN", "RN"]:
            edge_agent = EdgeAgentNoCacheV5(edge_config, task_config)
        else:
            raise NotImplementedError(method)

        if method == "OJOF":
            dev_bandit_params = {mycst.MAB_I_LINUCB: {"alpha": 0.01, "df_reward": 0.001, }, }
        elif method in ["LP", "LN"]:
            dev_bandit_params = {mycst.OFFLOAD_LOCAL: None, }
        elif method in ["EP", "EN"]:
            dev_bandit_params = {mycst.OFFLOAD_REMOTE: None, }
        elif method in ["RP", "RN"]:
            dev_bandit_params = {mycst.OFFLOAD_RANDOM: None, }
        elif method == "GP":
            dev_bandit_params = {mycst.OFFLOAD_GREEDY: None, }
        else:
            raise NotImplementedError
        dev_agent_list = [DeviceAgentV5(dev, task_config, dev_bandit_params) for dev in dev_configs]

        offload_and_cache(req_tids_seq, edge_agent, dev_agent_list)
        # offload metrics
        slot_cum_d_hist_dict, _ = get_offload_metric(dev_agent_list)
        fw_scd_hists.append(list(slot_cum_d_hist_dict.values())[0])
        # cache metrics
        chs_hist_dict, schs_hist_dict = get_cache_metric({method: edge_agent, })
        fw_chs_hists.append(list(chs_hist_dict.values())[0])
        fw_schs_hists.append(list(schs_hist_dict.values())[0])

    myeps.save_plot_figure(fw_methods, fw_scd_hists, "slots", "average delay (s)", f"{log_dir}/fw-scd-{T}-{N}-{M}", ncol=3)
    myeps.save_plot_figure(fw_methods, fw_chs_hists, "slots", "hit size (Mbit)", f"{log_dir}/fw-chs-{T}-{N}-{M}",
                           ncol=3)
    myeps.save_plot_figure(fw_methods, fw_schs_hists, "slots", "average hit size (Mbit)", f"{log_dir}/fw-schs-{T}-{N}-{M}",
                           ncol=3)
    save_data(f"{log_dir}/fw-{T}-{N}-{M}.npz", dev_configs, task_config, req_tids_seq, cp_config)

    fw_metrics = {}
    for i in range(len(fw_methods)):
        fw_metrics[fw_methods[i]] = [len(dev_configs), fw_scd_hists[i][-1], fw_chs_hists[i][-1], fw_schs_hists[i][-1]]
        print("scd, chs, schs: ", fw_methods[i], fw_metrics[fw_methods[i]])
    return fw_metrics


def test_fw_diff_devs(fw_methods, T, N_list, M, cache_ratio, zipf_params, data_files=None, log_dir="./logs/",
                      seed=None):
    """ test the framework under different devices """
    total_metrics = None
    for i in range(len(N_list)):
        N = N_list[i]
        data_file = data_files[i] if data_files is not None else None
        fw_metrics = test_framework_methods(fw_methods, T, N, M, cache_ratio, zipf_params, data_file, log_dir,
                                            seed=seed)
        if total_metrics is None:
            # total_metrics = fw_metrics
            total_metrics = {}
            for k, v in fw_metrics.items():
                total_metrics[k] = np.array(v).reshape((1, -1)) / N
        else:
            for k, v in total_metrics.items():
                # total_metrics[k] = v + fw_metrics[k]
                N_v = np.array(fw_metrics[k]).reshape((1, -1)) / N
                total_metrics[k] = np.concatenate((v, N_v))
    # plot metrics
    scd_list, chs_list, schs_list = [], [], []
    for fw in fw_methods:
        # val = np.reshape(total_metrics[fw], (len(N_list), 4))
        val = total_metrics[fw]
        scd_list.append(val[:, 1])
        chs_list.append(val[:, 2])
        schs_list.append(val[:, 3])
    N_labels = [f"{N}" for N in N_list]
    myeps.save_diff_bar_figure(N_labels, fw_methods, scd_list, "the number of users", "average delay (s)",
                               f"{log_dir}/fw-diff-devs-scd", ylim_min=0.5, figsize=(5.0, 3.0), ncol=4)
    # myeps.save_diff_bar_figure(N_labels, fw_methods, chs_list, "the number of users", "hit size (Mb)",
    #                            f"{log_dir}/fw-diff-devs-chs", ylim_min=0.0, figsize=(5.0, 3.0), ncol=4)
    # myeps.save_diff_bar_figure(N_labels, fw_methods, schs_list, "the number of users", "hit size (Mb)",
    #                            f"{log_dir}/fw-diff-devs-schs", ylim_min=0.0, figsize=(5.0, 3.0), ncol=4)


def test_fw_diff_tasks(fw_methods, T, N, M_list, cache_ratio, zipf_params, data_files=None, log_dir="./logs/",
                       seed=None):
    total_metrics = None
    for i in range(len(M_list)):
        M = M_list[i]
        data_file = data_files[i] if data_files is not None else None
        fw_metrics = test_framework_methods(fw_methods, T, N, M, cache_ratio, zipf_params, data_file, log_dir,
                                            seed=seed)
        if total_metrics is None:
            total_metrics = fw_metrics
        else:
            for k, v in total_metrics.items():
                total_metrics[k] = v + fw_metrics[k]
    # plot metrics
    scd_list, chs_list, schs_list = [], [], []
    for fw in fw_methods:
        val = np.reshape(total_metrics[fw], (len(M_list), 4))
        scd_list.append(val[:, 1] / N)
        chs_list.append(val[:, 2])
        schs_list.append(val[:, 3])
    M_labels = [f"{M}" for M in M_list]
    myeps.save_diff_bar_figure(M_labels, fw_methods, scd_list, "the number of tasks", "average delay (s)",
                               f"{log_dir}/fw-diff-tasks-scd", ylim_min=0.5, figsize=(5.0, 3.0), ncol=4)
    # myeps.save_diff_bar_figure(M_labels, fw_methods, chs_list, "the number of tasks", "hit size (Mbit)",
    #                            f"{log_dir}/fw-diff-tasks-chs", ylim_min=0.0, figsize=(5.0, 3.0), ncol=4)
    # myeps.save_diff_bar_figure(M_labels, fw_methods, schs_list, "the number of tasks", "average hit size (Mbit)",
    #                            f"{log_dir}/fw-diff-tasks-schs", ylim_min=0.0, figsize=(5.0, 3.0), ncol=4)


def test_offload_methods(o_methods, T, N, M, cache_ratio, zipf_params, data_file=None, log_dir="./logs/", seed=None,
                         save_dev_metric=False):
    # build edge and device agents
    bandit_params = {}
    for o in o_methods:
        bandit_params[o] = mycst.DEFAULT_BANDIT_PARAMS.get(o, None)

    if data_file is None:
        np.random.seed(seed)
        task_config = TaskConfig(M, cache_ratio)
        dev_configs = [DeviceConfig(T) for _ in range(N)]
        req_tids_seq, cp_config = generate_req_tids_seq(T, N, M, zipf_params, pattern="ROSS")
    else:
        dev_configs, task_config, req_tids_seq, cp_config = load_data(data_file, cache_ratio)

    dev_agent_list = [DeviceAgentV5(dev_config, task_config, bandit_params) for dev_config in dev_configs]
    edge_agent = EdgeAgentV5(EdgeConfig(task_config.max_cache_size), task_config,
                             mycst.DEFAULT_CA_PARAMS[mycst.CA_TLCPD])
    offload_and_cache(req_tids_seq, edge_agent, dev_agent_list)

    # 统计每个设备的结果
    if save_dev_metric:
        subpath = f"each-dev-{cache_ratio}"
        for n in range(req_tids_seq.shape[1]):
            cum_delay_hists, cum_regret_hists, suc_ratio_hists = dev_agent_list[n].get_metrics()
            myeps.save_plot_figure(list(cum_delay_hists.keys()), list(cum_delay_hists.values()), "slots",
                                   "cumulative delays (s)", f"{log_dir}/{subpath}/{n}-o-cum_d-hist")
            myeps.save_plot_figure(list(cum_regret_hists.keys()), list(cum_regret_hists.values()), "slots",
                                   "cumulative regrets (s)", f"{log_dir}/{subpath}/{n}-o-cum_r-hist")
            myeps.save_plot_figure(list(suc_ratio_hists.keys()), list(suc_ratio_hists.values()), "slots",
                                   "success arm ratios", f"{log_dir}/{subpath}/{n}-o-suc_ratio-hist")

    # 统计所有设备的总时延、总遗憾 cum_d_hist_dict, cum_r_hist_dict
    scd_hist_dict, cum_r_hist_dict = get_offload_metric(dev_agent_list)
    if cum_r_hist_dict is not None:
        myeps.save_plot_figure(list(cum_r_hist_dict.keys()), list(cum_r_hist_dict.values()), "slots",
                               "cumulative regret (s)",
                               f"{log_dir}/o-cr-{cache_ratio}", ncol=2)
    myeps.save_plot_figure(list(scd_hist_dict.keys()), list(scd_hist_dict.values()), "slots",
                           "average delay (s)",
                           f"{log_dir}/o-scd-{cache_ratio}", ncol=2)

    if not os.path.exists(f"{log_dir}/offload-{T}-{N}-{M}.npz"):
        save_data(f"{log_dir}/offload-{T}-{N}-{M}.npz", dev_configs, task_config, req_tids_seq, cp_config)
    o_metrics = {}
    for o in o_methods:
        o_metrics[o] = [cache_ratio, scd_hist_dict[o][-1], cum_r_hist_dict[o][-1]]
        print("cr, scd, cum_r: ", o, o_metrics[o])
    return o_metrics


def test_o_diff_caches(o_methods, T, N, M, cr_list, zipf_params, data_file=None, log_dir="./logs/", seed=None):
    total_metrics = None
    for cr in cr_list:
        o_metrics = test_offload_methods(o_methods, T, N, M, cr, zipf_params, data_file, log_dir, seed=seed)
        if total_metrics is None:
            total_metrics = o_metrics
        else:
            for k, v in total_metrics.items():
                total_metrics[k] = v + o_metrics[k]
    # plot metrics
    scd_list = []
    cum_r_list = []
    for o in o_methods:
        val = np.reshape(total_metrics[o], (len(cr_list), 3))
        scd_list.append(val[:, 1] / N)
        cum_r_list.append(val[:, 2])

    cr_labels = [f"{cr}" for cr in cr_list]
    myeps.save_diff_bar_figure(cr_labels, o_methods, scd_list, "cache ratio", "average delay (s)",
                               f"{log_dir}/o-diff-cr-scd", ylim_min=1.1, figsize=(5.0, 3.0), ncol=3)
    myeps.save_diff_bar_figure(cr_labels, o_methods, cum_r_list, "cache ratio", "cumulative regret (s)",
                               f"{log_dir}/o-diff-cr-cum_r", ylim_min=None)


def test_cache_methods(c_methods, T, N, M, cache_ratio, zipf_params, data_file=None, log_dir="./logs/", seed=None,
                       pattern="ROSS", figsize=(3.5, 3.5), plot_change_point=False):
    if data_file is None:
        np.random.seed(seed)
        task_config = TaskConfig(M, cache_ratio)
        dev_configs = [DeviceConfig(T) for _ in range(N)]
        req_tids_seq, cp_config = generate_req_tids_seq(T, N, M, zipf_params, pattern=pattern)
    else:
        dev_configs, task_config, req_tids_seq, cp_config = load_data(data_file, cache_ratio)

    edge_agent_dict = {}
    scd_hists = []
    # run each cache method
    for method in c_methods:
        edge_config = EdgeConfig(task_config.max_cache_size)
        if method == mycst.CA_NO_CACHE:
            edge_agent = EdgeAgentNoCacheV5(edge_config, task_config)
        elif method == mycst.CA_PERFECT:
            edge_agent = EdgeAgentPerfectV5(edge_config, task_config, cp_config)
        else:
            edge_agent = EdgeAgentV5(edge_config, task_config, mycst.DEFAULT_CA_PARAMS[method])
        bandit_params = {mycst.MAB_I_LINUCB: mycst.DEFAULT_BANDIT_PARAMS[mycst.MAB_I_LINUCB], }
        dev_agent_list = [DeviceAgentV5(dev_config, task_config, bandit_params) for dev_config in dev_configs]

        offload_and_cache(req_tids_seq, edge_agent, dev_agent_list)
        scd_hist_dict, cum_r_hist_dict = get_offload_metric(dev_agent_list)
        scd_hists.append(scd_hist_dict[mycst.MAB_I_LINUCB])

        if plot_change_point:
            edge_agent.plot_change_points(log_dir, cp_config, figsize=(5, 3.5))
        edge_agent_dict[method] = edge_agent

    # plot metrics
    chs_hist_dict, schs_hist_dict = get_cache_metric(edge_agent_dict)
    myeps.save_plot_figure(list(chs_hist_dict.keys()), list(chs_hist_dict.values()), "slots",
                           "cumulative hit size (Mbit)",
                           f"{log_dir}/c-chs-{pattern}")
    myeps.save_plot_figure(list(schs_hist_dict.keys()), list(schs_hist_dict.values()), "slots",
                           "average hit size (Mbit)",
                           f"{log_dir}/c-schs-{pattern}", fig_size=figsize, ylim_min=15.0, ncol=2)
    myeps.save_plot_figure(c_methods, scd_hists, "slots", "average delay (s)", f"{log_dir}/c-scd-{pattern}")

    save_data(f"{log_dir}/cache-{T}-{N}-{M}-{pattern}.npz", dev_configs, task_config, req_tids_seq,
              cp_config)