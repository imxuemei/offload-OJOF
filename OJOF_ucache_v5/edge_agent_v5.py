# -*- coding:utf-8 -*-
"""
edge接收所有用户的请求，进行缓存决策，处理卸载请求。
缓存统计指标：命中率、节约的吞吐量
缓存目标：提高命中率，或，提高吞吐量
1 考虑改变点，2 全局缓存放置
"""

import matplotlib.pyplot as plt
import numpy as np

import OJOF_ucache_v5.env_config_v2 as mycfg
import OJOF_ucache_v5.env_constants_v2 as mycst
from OJOF_ucache_v5.mec_cpd.mvcpd_v2 import PHT_MVCPD, ADWIN_MVCPD, TSCPD_MVCPD


class EdgeAgentV5:

    def __init__(self, edge: mycfg.EdgeConfig, tasks: mycfg.TaskConfig, ca_params):
        self.edge = edge
        self.tasks = tasks
        self.ca_params = ca_params
        self.cache_list = []
        self.M = tasks.num_tasks
        self._is_initiated = False

    def _init_cpd(self, N):
        self._is_initiated = True
        self.N = N
        self.ca_type = self.ca_params["ca_type"]
        self.cpd_topK = min(self.M, self.ca_params.get("cpd_topK", 1))
        cpd_config = self.ca_params.get("cpd_config", None)
        if self.ca_type == mycst.CA_PHT:
            self.cpd_agent_list = [PHT_MVCPD(self.cpd_topK, threshold=cpd_config["threshold"]) for _ in range(self.N)]
        elif self.ca_type == mycst.CA_ADWIN:
            self.cpd_agent_list = [ADWIN_MVCPD(self.cpd_topK, delta=cpd_config["delta"]) for _ in range(self.N)]
        elif self.ca_type == mycst.CA_TLCPD:
            self.cpd_agent_list = [TSCPD_MVCPD(self.cpd_topK, cpd_config) for _ in range(self.N)]
        else:
            self.cpd_agent_list = None

        self.hit_num_hist_NT = [[] for _ in range(self.N)]  # N*T, 每个设备的缓存累计命中数量历史
        self.hit_size_hist_NT = [[] for _ in range(self.N)]  # N*T, 每个设备的缓存累计节约吞吐量历史

        # 全局流行度及滑动窗口流行度
        self.sw_sizes = mycst.CACHE_SW_SIZE + np.zeros(self.N, dtype=np.int32)
        self.cum_nums_hist = []  # T*N*M，每个时刻，每个用户，每个任务的累积使用数量
        self.cum_pops_hist = []  # T*N*M，每个时刻，每个用户，每个任务累积流行度
        self.sw_pops_hist = []  # T*N*M，每个时刻，每个用户，每个任务滑动窗口内流行度
        self.cpd_pops_hist = []  # T*N*M，每个slot，每个用户，每个任务的用于缓存置换的流行度
        # cpd_cps_hist=N*change points，每个slot，每个用户，改变点列表  每个用户，历史改变点。cp_step：第x次req_tid时改变。
        self.cpd_cps_hist = [[] for _ in range(self.N)]
        self.cpd_topK_tids = [None for _ in range(self.N)]  # N*cpd_topN，每个用户，top N个任务id

    def get_c2e_rate(self):
        return self.edge.c2e_rate

    def allocate_cpu(self, num_users):
        assert num_users > 0, "num_users > 0"
        return self.edge.total_cpu / num_users

    def get_cache_list(self) -> list:
        return self.cache_list

    def _update_pops_hist(self, req_tids: np.ndarray):
        # 1 更新累计使用数量cum_nums
        task_nums = np.zeros((self.N, self.M), dtype=np.int32)
        for n, m in enumerate(req_tids):
            task_nums[n, m] = 1
        if len(self.cum_nums_hist) > 0:
            task_nums += self.cum_nums_hist[-1]
        self.cum_nums_hist.append(task_nums)

        hist_size = len(self.cum_nums_hist)
        # 2 计算cum_pops, sw_pops
        cum_pops = self.cum_nums_hist[-1] / hist_size
        self.cum_pops_hist.append(cum_pops)

        sw_pops = np.zeros((self.N, self.M))
        for n in range(self.N):
            n_sw_size = self.sw_sizes[n]
            if hist_size <= n_sw_size:
                sw_pops[n, :] = cum_pops[n, :]
            else:
                sw_pops[n, :] = (self.cum_nums_hist[-1][n] - self.cum_nums_hist[-n_sw_size - 1][
                    n]) / n_sw_size
        self.sw_pops_hist.append(sw_pops)

        # 3 计算上次改变点之后的累积流行度 user_pops
        cpd_cum_pops = np.zeros((self.N, self.M))
        for n in range(self.N):
            n_cps = self.cpd_cps_hist[n]
            if len(n_cps) == 0:
                # 如果没有改变点，则取总累计值作为cpd cum
                cpd_cum_pops[n, :] = self.cum_pops_hist[-1][n, :]
            else:
                # 计算从 cp_steps - sw_size -1 开始算
                n_task_nums = self.cum_nums_hist[-1][n, :] - self.cum_nums_hist[n_cps[-1] - 1 - n_sw_size][n, :]
                cpd_cum_pops[n, :] = n_task_nums / (len(self.cum_nums_hist) - n_cps[-1] + n_sw_size)
        self.cpd_pops_hist.append(cpd_cum_pops)

    def _compute_current_pops(self) -> np.ndarray:
        cur_steps = len(self.cum_pops_hist)
        # 初始化窗口中，直接返回cum_pops
        if cur_steps <= max(self.sw_sizes):
            return self.cum_pops_hist[-1]

        if self.ca_type == mycst.CA_LFU:
            # 使用累积pops
            return self.cum_pops_hist[-1]
        elif self.ca_type == mycst.CA_W_LFU:
            # 使用滑动窗口pops
            return self.sw_pops_hist[-1]
        elif self.cpd_agent_list is not None:
            # 使用带CPD的pops
            for n in range(self.N):
                if self.cpd_topK_tids[n] is None:
                    # 取前一次累积流行度最高的top N个任务
                    self.cpd_topK_tids[n] = np.argsort(-self.cum_pops_hist[-2][n])[0:self.cpd_topK]
                short_values = [self.sw_pops_hist[-1][n][tid] for tid in self.cpd_topK_tids[n]]
                long_values = [self.cpd_pops_hist[-1][n][tid] for tid in self.cpd_topK_tids[n]]
                short_values, long_values = np.array(short_values), np.array(long_values)

                in_change, all_changes, ft_alarm_list = self.cpd_agent_list[n].change_point_test(short_values,
                                                                                                 long_values)

                # 如果有改变点，则更新cpd_cps, cpd_pops, cpd_topN_tids
                if in_change:
                    self.cpd_cps_hist[n].append(cur_steps)
                    self.cpd_pops_hist[-1][n] = np.copy(self.sw_pops_hist[-1][n])
                    self.cpd_topK_tids[n] = np.argsort(-self.sw_pops_hist[-1][n])[0:self.cpd_topK]

                    # print(cur_steps, "user: ",n, in_change, all_changes, ft_alarm_list)
            # CPD判断完成，返回最新数据
            return self.cpd_pops_hist[-1]
        else:
            raise ValueError(f"{self.ca_type} is not supported")

    def cache_placement(self, task_pops) -> list:
        sorted_tids = np.argsort(-task_pops)  # big to small tids
        new_cache_list, total_size = [], 0
        for tid in sorted_tids:
            c_size = self.tasks.get_cdata_size(tid)
            if total_size + c_size <= self.edge.max_cache_size:
                new_cache_list.append(tid)
                total_size += c_size
            if self.edge.max_cache_size - total_size < self.tasks.min_cdata_size:
                break

        return new_cache_list

    def receive_tasks(self, req_tids: np.ndarray) -> list:
        if not self._is_initiated:
            self._init_cpd(len(req_tids))

        # 1 statistic the requested tids, compute the last task pops
        self._update_pops_hist(req_tids)
        cur_pops = self._compute_current_pops()
        sum_pops = np.sum(cur_pops, axis=0)  # size=(M,)
        # 2 cache replacement
        self.cache_list = self.cache_placement(sum_pops)
        return self.cache_list

    def update_hit_stat(self, req_tids: np.ndarray):
        if not self._is_initiated:
            self._init_cpd(req_tids.shape[0])

        for n in range(self.N):
            m = req_tids[n]
            hit_num, hit_size = 0, 0
            if m in self.cache_list:
                hit_num = 1
                hit_size = self.tasks.get_cdata_size(m)
            self.hit_num_hist_NT[n].append(hit_num)
            self.hit_size_hist_NT[n].append(hit_size)

    def get_metrics(self):
        # 只看所有用户的总命中率历史，总吞吐量历史
        hn_hist_T = np.sum(np.array(self.hit_num_hist_NT), axis=0)
        chn_hist = np.cumsum(hn_hist_T)
        schn_hist = chn_hist / np.arange(1, 1 + chn_hist.shape[0])

        hs_hist_T = np.sum(np.array(self.hit_size_hist_NT), axis=0)
        chs_hist = np.cumsum(hs_hist_T)
        schs_hist = chs_hist / np.arange(1, 1 + chs_hist.shape[0])
        return chs_hist, schs_hist, chn_hist, schn_hist

    def plot_change_points(self, path, cp_config, figsize=(5, 3.5)):
        if self.ca_type not in [mycst.CA_PHT, mycst.CA_ADWIN, mycst.CA_TLCPD]:
            return
        mycfg.make_dirs(path)
        perf_cps = cp_config["cp_steps"]
        perf_cp_pops = cp_config["cp_pops"]
        perf_pops_hist = np.ones((len(self.cpd_pops_hist), self.N, self.M))
        for n in range(self.N):
            for i in range(len(perf_cp_pops[n])):
                scp = 0 if i == 0 else perf_cps[n][i - 1]
                ecp = perf_cps[n][i]
                perf_pops_hist[scp:ecp, n, :] = perf_pops_hist[scp:ecp, n, :] * perf_cp_pops[n][i]

        topK_tids = np.argsort(-perf_pops_hist[0, n, :])[0:self.cpd_topK]
        num_axs = 3  # 4
        for n in range(self.N):
            n_max_pop = 0.35  # np.max(perf_cp_pops[n]) * 1.0
            fig, ax = plt.subplots(num_axs, 1, figsize=figsize)
            colors = ['#661D98', '#2CBDFE', '#F5B14C', '#47DBCD', '#F3A0F2']
            i = 0
            for tid in topK_tids:
                ax[0].plot(np.array(self.cpd_pops_hist)[:, n, tid], label=f"task{tid}", linewidth=1.0, color=colors[i])
                ax[1].plot(perf_pops_hist[:, n, tid], label=f"task{tid}", linewidth=1.0, color=colors[i])
                ax[2].plot(np.array(self.sw_pops_hist)[:, n, tid], label=f"task{tid}", linewidth=1.0, color=colors[i])
                i += 1
                # ax[3].plot(np.array(self.cum_pops_hist)[:, n, tid], label=f"task{tid}")

            for i in range(num_axs):
                ax[i].legend(fontsize=8)
                ax[i].set_ylim(0, n_max_pop)
            ax[0].set_ylabel("cp-\npopularity", fontsize=10)
            ax[1].set_ylabel("perfect\npopularity", fontsize=10)
            ax[2].set_ylabel("sw-\npopularity", fontsize=10)
            ax[2].set_xlabel("slots", fontsize=10)

            for cpd_cp in self.cpd_cps_hist[n]:
                ax[0].plot([cpd_cp, cpd_cp], [0, 1], color="r", linewidth=0.5)
            for perf_cp in perf_cps[n][0:-1]:
                ax[0].plot([perf_cp, perf_cp], [0, 1], color='g', linewidth=0.5)

            plt.tight_layout()
            plt.savefig(f"{path}/{self.ca_type}-user{n}.eps", format="eps", bbox_inches='tight', dpi=300)
            plt.close(fig)


class EdgeAgentPerfectV5(EdgeAgentV5):
    def __init__(self, edge: mycfg.EdgeConfig, tasks: mycfg.TaskConfig, cp_config: dict):
        super(EdgeAgentPerfectV5, self).__init__(edge, tasks, ca_params={"ca_type": mycst.CA_PERFECT, })
        self.cp_steps_2d = cp_config["cp_steps"]  # (N,num_cps)
        self.cp_pops_3d = cp_config["cp_pops"]  # (N, num_cps, M)
        self.cur_num_cps = np.zeros(len(self.cp_steps_2d), dtype=np.int)

    def receive_tasks(self, req_tids: np.ndarray):
        if not self._is_initiated:
            self._init_cpd(len(req_tids))
        # 1 统计当前收到的req_tids
        self._update_pops_hist(req_tids)
        # 2 求下一个时刻的任务流行度，进行缓存置换
        cur_step = len(self.cum_pops_hist)
        cur_pops = np.zeros((self.N, self.M))
        for n in range(self.N):
            cur_pops[n] = self.cp_pops_3d[n][self.cur_num_cps[n]]
            indices = np.where(self.cp_steps_2d[n] == cur_step)[0]
            if len(indices) > 0:
                self.cur_num_cps[n] = indices[0] + 1
        # 3 执行缓存置换
        sum_pops = np.sum(cur_pops, axis=0)  # 每个任务累加后流行度，M维
        self.cache_list = self.cache_placement(sum_pops)
        return self.cache_list


class EdgeAgentNoCacheV5(EdgeAgentV5):

    def __init__(self, edge: mycfg.EdgeConfig, tasks: mycfg.TaskConfig):
        super(EdgeAgentNoCacheV5, self).__init__(edge, tasks, ca_params={"ca_type": mycst.CA_NO_CACHE, })

    def receive_tasks(self, req_tids: np.ndarray):
        if not self._is_initiated:
            self._init_cpd(len(req_tids))
        # self._update_pops_hist(req_tids)
        self.cache_list = []
        return self.cache_list
