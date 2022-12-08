"""
Two-stage Change Point Detection for Task Cache.
coding: utf-8
"""
import numpy as np
import scipy.stats as scipyss
from numpy import seterr

seterr(all='raise')


class PHT_V2:
    def __init__(self, threshold=1.0, std_size=100):
        self.threshold = threshold
        self.std_size = std_size
        self.v_mean = 0.0
        self.v_count = 0
        self.err_sum = 0.0
        self.min_err_sum = None
        self.max_err_sum = None
        self.v_list = []

    def reset(self):
        self.v_mean = 0.0
        self.v_count = 0
        self.err_sum = 0.0
        self.min_err_sum = None
        self.max_err_sum = None

    def reset_sum(self):
        self.err_sum = 0.0
        self.min_err_sum = None
        self.max_err_sum = None

    def add_value(self, value, in_alarm_change):
        if in_alarm_change:
            self.reset()
        # 更新 value
        self.v_count += 1
        self.v_mean += (value - self.v_mean) / self.v_count

    def test_alarm(self, value, thd_var_num=None) -> bool:
        self.v_list.append(value)
        # 需要实际更新error_sum以保证反映短期数据的变化
        v_mean = self.v_mean
        v_mean += (value - v_mean) / (self.v_count + 1)
        # 更新 err_sum
        self.err_sum = 0.9999 * self.err_sum + value - v_mean
        if self.min_err_sum is None or self.min_err_sum > self.err_sum:
            self.min_err_sum = self.err_sum
        if self.max_err_sum is None or self.max_err_sum < self.err_sum:
            self.max_err_sum = self.err_sum

        if thd_var_num is None or len(self.v_list) <= self.std_size:
            threshold = self.threshold
        else:
            v_std = np.std(self.v_list)
            threshold = thd_var_num * v_std if v_std > 0.0 else self.threshold

        in_alarm_change = False
        # 判断递增超过阈值, 递减超过阈值
        if (self.err_sum - self.min_err_sum > threshold) or (self.max_err_sum - self.err_sum > threshold):
            in_alarm_change = True
        print("pht %.4f, %.4f, %.4f, %.4f" % (
            value, threshold, self.err_sum - self.min_err_sum, self.max_err_sum - self.err_sum))
        return in_alarm_change


class PHT_V3:
    def __init__(self, threshold=1.0, init_size=5):
        self.threshold = threshold
        self.init_size = init_size
        self.v_count, self.v_mean = 0, 0.0
        self.alrm_v_list = []
        self.in_sum, self.de_sum = 0.0, 0.0
        self.max_sum, self.min_sum = 0.0, 0.0

    def reset(self):
        self.v_count, self.v_mean = 0, 0.0
        self.alrm_v_list = []
        self.in_sum, self.de_sum = 0.0, 0.0
        self.max_sum, self.min_sum = 0.0, 0.0

    def add_value(self, value, in_alarm_change):
        if in_alarm_change:
            self.reset()
        self.v_count += 1
        self.v_mean = self.v_mean + (value - self.v_mean) / self.v_count

    def test_alarm(self, value, thd_std_num=None, is_log = False) -> bool:
        self.alrm_v_list.append(value)
        v_std = np.std(self.alrm_v_list) if len(self.alrm_v_list) >= self.init_size else 0.0
        v_mean = value if self.v_count == 0 else self.v_mean
        threshold = thd_std_num * v_std if (v_std > 0 and thd_std_num is not None) else self.threshold

        self.in_sum = self.in_sum + value - v_mean - v_std
        self.min_sum = min(self.min_sum, self.in_sum)
        increase_alarm = self.in_sum - self.min_sum > threshold

        self.de_sum = self.de_sum + value - v_mean + v_std
        self.max_sum = max(self.max_sum, self.de_sum)
        decrease_alarm = self.max_sum - self.de_sum > threshold

        in_change = increase_alarm or decrease_alarm
        if is_log:
            print("pht %s, %.4f, %.4f, %.4f, %.4f" % (in_change, threshold, self.in_sum - self.min_sum, self.max_sum-self.de_sum, value))
        return in_change

def entropy_clean_zero(p: np.ndarray, q: np.ndarray):
    np, nq = [], []
    for i in range(p.shape[0]):
        if p[i] > 0 and q[i] > 0:
            np.append(p[i])
            nq.append(q[i])
    enp = scipyss.entropy(np, nq) if len(np) > 0 else 0.0
    return enp


class TSCPD_V2:

    def __init__(self, num_features, tscpd_config):
        self.num_features = num_features
        self.ft_pht_threshold = tscpd_config.get("ft_pht_threshold", 5.0)
        self.ft_pht_tv_num = tscpd_config.get("ft_pht_tv_num", 100)
        self.cp_cluster_ws = tscpd_config.get("cp_cluster_ws", 50)
        self.kl_pht_threshold = tscpd_config.get("kl_pht_threshold", 5.0)
        self.kl_pht_tv_num = tscpd_config.get("kl_pht_tv_num", 200)
        # self.kl_threshold = tscpd_config.get("kl_threshold") #KL不使用PHT，直接判断的情况

        self.feature_pht_list = [PHT_V3(self.ft_pht_threshold) for _ in range(self.num_features)]
        self.kl_pht = PHT_V3(self.kl_pht_threshold)
        self.cur_step = 0
        self.ft_cp_steps = np.zeros(self.num_features, dtype=np.int32) - self.cp_cluster_ws
        self.kl_cp_step = -self.cp_cluster_ws
        self.order_cp_step = -self.cp_cluster_ws

    def reset(self):
        for ft_pht in self.feature_pht_list:
            ft_pht.reset()
        self.kl_pht.reset()
        self.cur_step = 0
        self.ft_cp_steps = np.zeros(self.num_features, dtype=np.int32) - self.cp_cluster_ws
        self.kl_cp_step = -self.cp_cluster_ws

    def update(self, short_values: np.ndarray, long_values: np.ndarray) -> tuple:
        self.cur_step += 1
        if self.num_features == 1:
            return self._update_single_value(short_values, long_values)
        else:
            return self._update_multiple_values(short_values, long_values)

    def _update_single_value(self, short_values: np.ndarray, long_values: np.ndarray) -> tuple:
        assert self.num_features == 1, "only for 1 feature data"
        in_change = self.feature_pht_list[0].test_alarm(short_values[0], thd_std_num=self.ft_pht_tv_num)
        if in_change:
            self.reset()
        else:
            self.feature_pht_list[0].add_value(long_values[0], in_change)
        return in_change, [in_change], [in_change]

    def _update_multiple_values(self, short_values: np.ndarray, long_values: np.ndarray) -> tuple:
        assert self.num_features > 1, "only for more than one feature data"
        all_changes = []
        # 判断每维特征是否改变
        ft_alarm, ft_alarm_list = False, []
        for i in range(self.num_features):
            in_alarm = self.feature_pht_list[i].test_alarm(short_values[i], thd_std_num=self.ft_pht_tv_num)
            ft_alarm_list.append(in_alarm)
            if in_alarm:
                ft_alarm = True
                self.ft_cp_steps[i] = self.cur_step
        # 如果有警告，那么采用3种方式判断是否为change point
        # 1 判断改变点是否聚集 todo 最简单的方式就是与当前step距离最近的点的个数
        cluster_num = 0
        for pos in (self.ft_cp_steps - self.cur_step + self.cp_cluster_ws):
            if pos >= 0:
                cluster_num += 1
        # 如果当前没有改变点，那么不判断聚集
        cluster_change = (cluster_num >= self.num_features) if ft_alarm else False
        all_changes.append(cluster_change)

        # 2 判断KL的改变 todo 可以直接使用threshold判断，也可以使用PHT判断
        kl = entropy_clean_zero(long_values, short_values)
        # kl_pht_change = self.kl_pht.test_alarm(abs(kl), thd_std_num=self.kl_pht_tv_num, is_log=True)
        kl_pht_change = (kl > 0.1)
        if kl_pht_change:
            self.kl_cp_step = self.cur_step
        kl_change = (self.kl_cp_step - self.cur_step + self.cp_cluster_ws >= 0)
        all_changes.append(kl_change)
        # all_changes.append(kl_change)

        # 3 判断数值顺序 与 改变前是否有连续变化
        order_cur_change = False
        long_order = np.argsort(long_values)
        short_order = np.argsort(short_values)
        for i in range(self.num_features):
            if long_order[i] != short_order[i]:
                order_cur_change = True
                break
        not_freq_change = True
        if order_cur_change:
            not_freq_change = self.cur_step - self.order_cp_step > 100
            self.order_cp_step = self.cur_step
        order_change = (self.order_cp_step - self.cur_step + self.cp_cluster_ws >= 0) if not_freq_change else False
        all_changes.append(order_change)

        # 4 如果有3个以上判断为change，则认为change
        # in_change = len([x for x in all_changes if x]) > 2
        in_change = cluster_change
        if in_change:
            # 如果改变，直接重置所有PHT，因为topK会改变
            self.reset()
        else:
            # todo 也可以按照各自真实的改变更新
            for i in range(self.num_features):
                self.feature_pht_list[i].add_value(long_values[i], ft_alarm_list[i])
            self.kl_pht.add_value(kl, kl_pht_change)

        return in_change, all_changes, ft_alarm_list
