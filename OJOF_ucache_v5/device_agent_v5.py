# -*- coding:utf-8 -*-
import numpy as np

import OJOF_ucache_v5.env_config_v2 as mycfg
import OJOF_ucache_v5.env_constants_v2 as mycst
import OJOF_ucache_v5.mec_bandit.bandit_models_v2 as bm


class DeviceAgentV5:

    def __init__(self, device: mycfg.DeviceConfig, tasks: mycfg.TaskConfig, bandit_params: dict):
        self.device = device
        self.tasks = tasks
        self.bandit_dict = self._init_bandit_dict(bandit_params)

    def _init_bandit_dict(self, bandit_params):
        """
        generate all offload method agents
        """
        num_features = 4
        bandit_dict = {}
        for type in bandit_params.keys():
            bandit_config = bandit_params.get(type, {})
            if type == mycst.MAB_I_LINUCB:
                bandit_dict[type] = bm.ILinucbBandit(num_features, bandit_config["alpha"],
                                                     bandit_config["df_reward"])
            elif type == mycst.MAB_LINUCB:
                bandit_dict[type] = bm.LinucbBandit(num_features, bandit_config["alpha"])
            elif type == mycst.MAB_E_GREEDY:
                bandit_dict[type] = bm.EgreedyBandit(bandit_config["epsilon"])
            elif type == mycst.MAB_UCB1:
                bandit_dict[type] = bm.Ucb1Bandit()
            elif type == mycst.OFFLOAD_GREEDY:
                bandit_dict[type] = bm.BaseBandit()
            elif type == mycst.OFFLOAD_LOCAL:
                bandit_dict[type] = bm.BaseBandit()
            elif type == mycst.OFFLOAD_REMOTE:
                bandit_dict[type] = bm.BaseBandit()
            elif type == mycst.OFFLOAD_RANDOM:
                bandit_dict[type] = bm.BaseBandit()
            else:
                raise NotImplementedError
        return bandit_dict

    def _delay2reward(self, delay):
        return - delay

    def _reward2delay(self, reward):
        return -reward

    def get_bandit_types(self) -> list:
        return list(self.bandit_dict.keys())

    def compute_local_delay(self, t, tid, cache_status, c2e_rate):
        cdata_c2e_t = self.tasks.comp_cdata_t(tid, c2e_rate) if cache_status == mycst.NO_CACHE else 0.0
        cdata_e2d_t = self.tasks.comp_cdata_t(tid, self.device.get_e2d_rate(t))
        exe_t = self.tasks.comp_exe_t(tid, self.device.cpu)
        local_delay = cdata_c2e_t + cdata_e2d_t + exe_t
        return local_delay

    def compute_remote_delay(self, t, tid, cache_status, c2e_rate, edge_cpu):
        cdata_c2e_t = self.tasks.comp_cdata_t(tid, c2e_rate) if cache_status == mycst.NO_CACHE else 0.0
        idata_d2e_t = self.tasks.comp_idata_t(tid, self.device.get_d2e_rate(t))
        exe_t = self.tasks.comp_exe_t(tid, edge_cpu)
        odata_e2d_t = self.tasks.comp_odata_t(tid, self.device.get_e2d_rate(t))
        remote_delay = max(cdata_c2e_t, idata_d2e_t) + exe_t + odata_e2d_t
        return remote_delay

    def select_arms(self, t, tid, cache_status, c2e_rate, avg_edge_cpu):
        """
        :return: arm_dict, local_delay, avg_remote_delay
        """
        local_delay = self.compute_local_delay(t, tid, cache_status, c2e_rate)
        avg_remote_delay = self.compute_remote_delay(t, tid, cache_status, c2e_rate, avg_edge_cpu)

        arm_dict, delay_dict = {}, {}
        for bandit_type in self.bandit_dict.keys():
            if bandit_type == mycst.OFFLOAD_LOCAL:
                arm = mycst.ARM_C_L if cache_status == mycst.CACHED else mycst.ARM_NOC_L
            elif bandit_type == mycst.OFFLOAD_REMOTE:
                arm = mycst.ARM_C_R if cache_status == mycst.CACHED else mycst.ARM_NOC_R
            elif bandit_type == mycst.OFFLOAD_RANDOM:
                if cache_status == mycst.CACHED:
                    arm = np.random.choice([mycst.ARM_C_L, mycst.ARM_C_R])
                else:
                    arm = np.random.choice([mycst.ARM_NOC_L, mycst.ARM_NOC_R])
            elif bandit_type == mycst.OFFLOAD_GREEDY:
                if local_delay <= avg_remote_delay:
                    arm = mycst.ARM_C_L if cache_status == mycst.CACHED else mycst.ARM_NOC_L
                else:
                    arm = mycst.ARM_C_R if cache_status == mycst.CACHED else mycst.ARM_NOC_R
            else:
                context = {
                    "x": self.tasks.get_task(tid),
                    "cache_status": cache_status,
                }
                arm = self.bandit_dict[bandit_type].select_arm(context=context)
            arm_dict[bandit_type] = arm
        return arm_dict, local_delay, avg_remote_delay

    def pull_arms(self, tid, arm_dict, delay_dict):
        for bandit_type in self.bandit_dict.keys():
            arm, delay = arm_dict[bandit_type], delay_dict[bandit_type]
            context = {
                "x": self.tasks.get_task(tid),
            }
            reward = self._delay2reward(delay)
            self.bandit_dict[bandit_type].pull_arm(arm, reward, context=context)

    def get_one_bandit_metric(self, bandit_type, perf_arm_hist=None, perf_cum_reward_hist=None):
        """
        :return: 返回总时延、累积时延hist、累积遗憾hist、成功arm比例hist
        """
        # 查找perfect arm, reward
        if perf_arm_hist is None or perf_cum_reward_hist is None:
            perf_bandit = self.bandit_dict.get(mycst.OFFLOAD_GREEDY, None)
            if perf_bandit is not None:
                perf_arm_hist, perf_reward_hist = perf_bandit.get_arm_reward_hist()
                perf_cum_reward_hist = np.cumsum(perf_reward_hist)

        arm_hist, reward_hist = self.bandit_dict[bandit_type].get_arm_reward_hist()
        # 计算arm成功率
        T, suc_ratio_hist = len(arm_hist), None
        if perf_arm_hist is not None:
            suc_num, suc_num_hist = 0, []
            for t in range(T):
                suc_num += (1 if arm_hist[t] == perf_arm_hist[t] else 0)
                suc_num_hist.append(suc_num)
            suc_ratio_hist = np.array(suc_num_hist) / np.arange(1, T + 1)
        # 计算累积遗憾
        cum_regret_hist = None
        if perf_cum_reward_hist is not None:
            cum_regret_hist = perf_cum_reward_hist - np.cumsum(reward_hist)
        # 计算累积时延和总时延
        cum_delay, cum_delay_hist = 0, []
        for r in reward_hist:
            cum_delay += self._reward2delay(r)
            cum_delay_hist.append(cum_delay)

        return cum_delay_hist, cum_regret_hist, suc_ratio_hist

    def get_metrics(self):
        """
        :return: 返回每个bandit的总时延、累积时延hist、累积遗憾hist、成功arm比例hist
        """
        cum_delay_hists, cum_regret_hists, suc_ratio_hists = {}, {}, {}
        p_a_hist, p_cum_r_hist = None, None
        perf_bandit = self.bandit_dict.get(mycst.OFFLOAD_GREEDY, None)
        if perf_bandit is not None:
            p_a_hist, p_r_hist = perf_bandit.get_arm_reward_hist()
            p_cum_r_hist = np.cumsum(p_r_hist)

        for bt in self.bandit_dict.keys():
            cum_delay_hists[bt], cum_regret_hists[bt], suc_ratio_hists[bt] = self.get_one_bandit_metric(bt,
                                                                                                        perf_arm_hist=p_a_hist,
                                                                                                        perf_cum_reward_hist=p_cum_r_hist)
        if perf_bandit is None:
            return cum_delay_hists, None, None
        else:
            return cum_delay_hists, cum_regret_hists, suc_ratio_hists
