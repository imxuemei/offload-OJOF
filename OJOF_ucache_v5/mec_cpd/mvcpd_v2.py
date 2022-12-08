"""
topN由edge处理，这里仅处理多维数据改变点
encoding: utf-8
"""
import numpy as np
from river.drift.adwin import ADWIN
from river.drift.page_hinkley import PageHinkley
from OJOF_ucache_v5.mec_cpd.tscpd_v2 import TSCPD_V2

class BaseMVCPD:

    def change_point_test(self, short_values: np.ndarray, long_values: np.ndarray) -> tuple:
        """
        检测改变点
        :param short_values: 用于测试是否改变的数值
        :param long_values: 用于实际取平均的数值
        :return: (in_change, ft_alarm_list) in_change: 如果改变则返回True，否则返回False。ft_alarm_list：警告列表。
        """
        raise NotImplementedError


class PHT_MVCPD(BaseMVCPD):

    def __init__(self, num_features, threshold):
        self.num_features = num_features
        self.ft_pht_list = [PageHinkley(threshold=threshold) for _ in range(self.num_features)]

    def change_point_test(self, short_values: np.ndarray, long_values: np.ndarray) -> tuple:
        ft_change_list = []
        for i in range(self.num_features):
            i_in_change, _ = self.ft_pht_list[i].update(short_values[i])
            ft_change_list.append(i_in_change)
        return all(ft_change_list), ft_change_list


class ADWIN_MVCPD(BaseMVCPD):
    def __init__(self, num_features, delta):
        self.num_features = num_features
        self.ft_pht_list = [ADWIN(delta=delta) for _ in range(self.num_features)]

    def change_point_test(self, short_values: np.ndarray, long_values: np.ndarray) -> tuple:
        ft_change_list = []
        for i in range(self.num_features):
            i_in_change, _ = self.ft_pht_list[i].update(short_values[i])
            ft_change_list.append(i_in_change)
        return all(ft_change_list), ft_change_list


class TSCPD_MVCPD(BaseMVCPD):

    def __init__(self, num_features, cpd_config):
        self.tscpd = TSCPD_V2(num_features, tscpd_config=cpd_config)

    def change_point_test(self, short_values: np.ndarray, long_values: np.ndarray) -> tuple:
        return self.tscpd.update(short_values, long_values)