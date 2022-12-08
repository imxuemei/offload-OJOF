# -*- coding:utf-8 -*-
"""
Device在未知信息下，自主决策是否进行任务卸载。
"""
import numpy as np

import OJOF_ucache_v5.env_constants_v2 as mycst
from OJOF_ucache_v5.mec_bandit.online_rr import OnlineRidgeRegression


class BaseBandit:
    """
    Collect the pulled arm and reward for the statistics.
    """

    def __init__(self):
        self.reward_hist = []
        self.arm_hist = []

    def select_arm(self, context) -> int:
        raise NotImplementedError

    def pull_arm(self, arm, reward, context=None):
        self.arm_hist.append(arm)
        self.reward_hist.append(reward)

    def get_arm_reward_hist(self):
        return self.arm_hist, self.reward_hist

    def encode_arm(self, cache_status, offload_status):
        if cache_status == mycst.CACHED:
            return mycst.ARM_C_R if offload_status == mycst.REMOTE else mycst.ARM_C_L
        else:
            return mycst.ARM_NOC_R if offload_status == mycst.REMOTE else mycst.ARM_NOC_L

    def decode_arm(self, arm):
        if arm == mycst.ARM_C_L:
            cache_status, offload_status = mycst.CACHED, mycst.LOCAL
        elif arm == mycst.ARM_C_R:
            cache_status, offload_status = mycst.CACHED, mycst.REMOTE
        elif arm == mycst.ARM_NOC_L:
            cache_status, offload_status = mycst.NO_CACHE, mycst.LOCAL
        elif arm == mycst.ARM_NOC_R:
            cache_status, offload_status = mycst.NO_CACHE, mycst.REMOTE
        else:
            ValueError(f"no such arm {arm}")
        return cache_status, offload_status

    def _select_max_arm(self, cache_status, value_dict):
        if cache_status == mycst.CACHED:
            return mycst.ARM_C_L if value_dict[mycst.ARM_C_L] > value_dict[mycst.ARM_C_R] else mycst.ARM_C_R
        else:
            return mycst.ARM_NOC_L if value_dict[mycst.ARM_NOC_L] > value_dict[mycst.ARM_NOC_R] else mycst.ARM_NOC_R


class ILinucbBandit(BaseBandit):

    def __init__(self, num_features, alpha, df_reward):
        super().__init__()
        self.num_features = num_features
        self.alpha = alpha
        self.df_reward = df_reward
        self.orr_dict = {arm: OnlineRidgeRegression(num_features, alpha) for arm in mycst.ARM_LIST}
        self.arms_chosen_dict = {arm: 0 for arm in mycst.ARM_LIST}
        self.arm_init_num = 100
        self.arm_noc_r_reset = False

    # def get_pred_rewards(self, x):
    #     x = np.reshape(x, (self.num_features, 1))
    #     rew_dict = {}
    #     for arm, orr in self.orr_dict.items():
    #         rew_dict[arm] = orr.predict(x)
    #     return rew_dict

    def select_arm(self, context) -> int:
        x = np.reshape(context.get("x"), (self.num_features, 1))
        rew_dict = {}
        for arm, orr in self.orr_dict.items():
            rew_dict[arm] = orr.predict(x)
        # reset arm_noc_r predict
        if self.arms_chosen_dict[mycst.ARM_C_R] >= self.arm_init_num:
            rew_dict[mycst.ARM_NOC_R] = min(rew_dict[mycst.ARM_C_R], rew_dict[mycst.ARM_NOC_R])
        return self._select_max_arm(context.get("cache_status"), rew_dict)

    def pull_arm(self, arm, reward, context):
        self.arm_hist.append(arm)
        self.reward_hist.append(reward)
        self.arms_chosen_dict[arm] += 1

        if self.arms_chosen_dict[mycst.ARM_C_R] == self.arm_init_num and (not self.arm_noc_r_reset):
            self.orr_dict[mycst.ARM_NOC_R].reset()
            self.arm_noc_r_reset = True

        x = np.reshape(context.get("x"), (self.num_features, 1))
        if arm == mycst.ARM_NOC_R:
            # if there is no ARM_C_R, directly update ARM_NOC_R
            if not self.arm_noc_r_reset:
                self.orr_dict[arm].fit_one(x, reward)
            else:
                c_r = self.orr_dict[mycst.ARM_C_R].predict(x)
                noc_r = self.orr_dict[mycst.ARM_NOC_R].predict(x)
                # update ARM_NOC_R with the reward close to it
                if abs(noc_r - reward) <= abs(c_r - reward):
                    self.orr_dict[mycst.ARM_NOC_R].fit_one(x, reward)
        else:
            # update other three arms
            self.orr_dict[arm].fit_one(x, reward)


class LinucbBandit(BaseBandit):
    def __init__(self, num_features, alpha):
        super().__init__()
        self.num_features = num_features
        self.alpha = alpha
        self.orr_dict = {arm: OnlineRidgeRegression(num_features, alpha) for arm in mycst.ARM_LIST}

    # def get_pred_rewards(self, x):
    #     x = np.reshape(x, (self.num_features, 1))
    #     rew_dict = {}
    #     for arm, orr in self.orr_dict.items():
    #         rew_dict[arm] = orr.predict(x)
    #     return rew_dict

    def select_arm(self, context) -> int:
        x = np.reshape(context.get("x"), (self.num_features, 1))
        rew_dict = {}
        for arm, orr in self.orr_dict.items():
            rew_dict[arm] = orr.predict(x)
        return self._select_max_arm(context.get("cache_status"), rew_dict)

    def pull_arm(self, arm, reward, context):
        self.arm_hist.append(arm)
        self.reward_hist.append(reward)

        x = np.reshape(context.get("x"), (self.num_features, 1))
        self.orr_dict[arm].fit_one(x, reward)


class Ucb1Bandit(BaseBandit):

    def __init__(self):
        super().__init__()
        self.arms_chosen_dict = {arm: 0 for arm in mycst.ARM_LIST}
        self.arms_mean_r_dict = {arm: 0.0 for arm in mycst.ARM_LIST}
        self.max_ucb = 1e10

    def select_arm(self, context) -> int:
        cache_status = context.get("cache_status")
        if cache_status == mycst.CACHED:
            a_list = [mycst.ARM_C_L, mycst.ARM_C_R]
        else:
            a_list = [mycst.ARM_NOC_L, mycst.ARM_NOC_R]
        total_num = np.sum([self.arms_chosen_dict[a] for a in a_list])
        ucb_dict = {}
        for a in a_list:
            if self.arms_chosen_dict[a] == 0:
                ucb_dict[a] = self.max_ucb
            else:
                ucb_dict[a] = self.arms_mean_r_dict[a] + np.sqrt(2 * np.log(total_num) / self.arms_chosen_dict[a])
        # print("ucb1 mean_r ", self.arms_mean_r_dict)
        return self._select_max_arm(cache_status, ucb_dict)

    def pull_arm(self, arm, reward, context=None):
        self.arm_hist.append(arm)
        self.reward_hist.append(reward)

        self.arms_chosen_dict[arm] += 1
        mean_r = self.arms_mean_r_dict[arm]
        self.arms_mean_r_dict[arm] = mean_r + (reward - mean_r) / self.arms_chosen_dict[arm]


class EgreedyBandit(BaseBandit):
    def __init__(self, epsilon):
        super().__init__()
        self.arms_chosen_dict = {arm: 0 for arm in mycst.ARM_LIST}
        self.arms_mean_r_dict = {arm: 0.0 for arm in mycst.ARM_LIST}
        self.epsilon = epsilon

    def select_arm(self, context) -> int:
        cache_status = context.get("cache_status")
        if cache_status == mycst.CACHED:
            a_list = [mycst.ARM_C_L, mycst.ARM_C_R]
        else:
            a_list = [mycst.ARM_NOC_L, mycst.ARM_NOC_R]

        if np.random.random() <= self.epsilon:
            return np.random.choice(a_list)
        else:
            return self._select_max_arm(cache_status, self.arms_mean_r_dict)

    def pull_arm(self, arm, reward, context=None):
        self.arm_hist.append(arm)
        self.reward_hist.append(reward)

        self.arms_chosen_dict[arm] += 1
        mean_r = self.arms_mean_r_dict[arm]
        self.arms_mean_r_dict[arm] = mean_r + (reward - mean_r) / self.arms_chosen_dict[arm]
