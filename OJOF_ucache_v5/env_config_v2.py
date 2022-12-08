# -*- coding:utf-8 -*-
import os
import numpy as np


def make_dirs(url):
    fpath, fname = os.path.split(url)
    if not os.path.exists(fpath):
        os.makedirs(fpath)


def comp_rate(B_MHz, power_mW, d_m):
    """
    :param B_MHz: bandwidth, MHz
    :param power_mW: transmit power, milliwatt
    :param d_m: distance, meter
    :return:
    """

    def dB2num(x_dB):
        return pow(10.0, x_dB / 10.0)

    def num2dB(x_num):
        return 10.0 * np.log10(x_num)

    # free space path loss model
    pi = 3.14
    c = 3.0 * 1e8  # light speed, 3*10^8 m/s
    f = 915.0 * 1e6  # carrier, 915 MHz
    G = 4.11  # antenna gain, 4.11
    path_gain = ((G * np.square(c / f)) ** 2) / ((4 * pi * d_m) ** 2)

    # rayleigh fading
    h_t = np.sqrt(0.5) * (np.random.standard_normal() + 1j * np.random.standard_normal())
    rayleigh_gain_t = abs(h_t) ** 2

    # capacity computing
    B = B_MHz * 1e6
    sigma2_mW = dB2num(-174.0) * B  # -174dBm/Hz
    SNR = (power_mW * path_gain * rayleigh_gain_t) / sigma2_mW
    rate_Mbps = B * np.log10(1 + SNR) / 1e6
    return rate_Mbps


class DeviceConfig:

    def __init__(self, T):
        self.bandwidth = 1.0  # 10 MHz
        self.d2e_power = 500.0  # mW, uplink power
        self.e2d_power = 1000.0  # mW, downlink power
        self.T = T
        self.cpu = 0.1 * np.random.randint(5, 16)  # 0.5-1.5GHz
        self.distance = 10.0 * np.random.randint(3, 8)  # 30m-70m
        self.d2e_rates = [comp_rate(B_MHz=self.bandwidth, power_mW=self.d2e_power, d_m=self.distance) for _ in range(T)]
        self.e2d_rates = [comp_rate(B_MHz=self.bandwidth, power_mW=self.e2d_power, d_m=self.distance) for _ in range(T)]

    def get_d2e_rate(self, t):
        return self.d2e_rates[t]

    def get_e2d_rate(self, t):
        return self.e2d_rates[t]


class EdgeConfig:

    def __init__(self, max_cache_size):
        self.total_cpu = 25.0  # 30 GHz, total cpu
        self.c2e_rate = 15.0  # Mbps
        self.e2d_power = 1000.0  # 1000mW
        self.max_cache_size = max_cache_size  # Mbit


class TaskConfig:
    def __init__(self, num_tasks, cache_ratio):
        self.num_tasks = num_tasks
        self.tasks = self._gen_tasks(num_tasks)
        self.min_cdata_size = np.min(self.tasks[:, 2])
        self.max_cdata_size = np.max(self.tasks[:, 2])
        self.max_cache_size = round(cache_ratio * np.sum(self.tasks[:, 2]), 3)

    def reset(self, cache_ratio, tasks: np.ndarray = None):
        if tasks is not None:
            self.tasks = tasks
            self.num_tasks = tasks.shape[0]
            self.min_cdata_size = np.min(self.tasks[:, 2])
            self.max_cdata_size = np.max(self.tasks[:, 2])
        self.max_cache_size = round(cache_ratio * np.sum(self.tasks[:, 2]), 3)

    def _gen_tasks(self, num_tasks: int) -> np.ndarray:
        """
        :param num_tasks: the number of tasks
        :return: tasks=[num_tasks, 4]
        """
        cpus = 0.1 * np.random.randint(5, 16, size=(num_tasks, 1))  # 0.5-1.5Gcycles, cpu reqs
        idata = 0.5 * np.random.randint(1, 11, size=(num_tasks, 1))  # 3-6Mbit, input data
        cdata = 0.5 * np.random.randint(6, 13, size=(num_tasks, 1))  # 0.5-5Mbit, cloud data
        odata = 0.5 * np.random.randint(6, 13, size=(num_tasks, 1))  # 3-6Mbit, output data
        tasks = np.concatenate((cpus, idata, cdata, odata), axis=-1)
        return tasks

    def get_task(self, tid):
        return self.tasks[tid]

    def get_cdata_size(self, tid):
        return self.tasks[tid, 2]

    def comp_cdata_t(self, tid, rate_Mbps):
        return self.tasks[tid, 2] / rate_Mbps

    def comp_idata_t(self, tid, rate_Mbps):
        return self.tasks[tid, 1] / rate_Mbps

    def comp_odata_t(self, tid, rate_Mbps):
        return self.tasks[tid, 3] / rate_Mbps

    def comp_exe_t(self, tid, cpu_GHz):
        return self.tasks[tid, 0] / cpu_GHz


def load_data(file, cache_ratio):
    # dev_configs, task_config
    assert file[-3:] == 'npz', "numpy data format"
    data = np.load(file)
    dev_array, task_array = data["devs"], data["tasks"]
    dev_configs = []
    for dev in dev_array:
        dcfg = DeviceConfig(1)
        dcfg.T, dcfg.cpu, dcfg.distance = int(dev[0]), dev[1], dev[2]
        dcfg.d2e_rates = list(dev[3: 3 + dcfg.T])
        dcfg.e2d_rates = list(dev[3 + dcfg.T:3 + 2 * dcfg.T])
        dev_configs.append(dcfg)
    # parse task
    task_config = TaskConfig(1, cache_ratio)
    task_config.reset(cache_ratio, tasks=task_array)
    req_tids_seq = data["reqs"]
    cp_config = {
        "cp_pops": data["cp_pops"],
        "cp_steps": data["cp_steps"],
    }
    return dev_configs, task_config, req_tids_seq, cp_config


def save_data(file, dev_configs, task_config, req_tids_seq, cp_config):
    assert file[-3:] == 'npz', "numpy data format"
    dev_array = []
    for dcfg in dev_configs:
        dev = [dcfg.T, dcfg.cpu, dcfg.distance] + dcfg.d2e_rates + dcfg.e2d_rates
        dev_array.append(dev)
    dev_array = np.array(dev_array)
    cp_pops = np.array(cp_config['cp_pops'])
    cp_steps = np.array(cp_config['cp_steps'])
    np.savez(file, devs=dev_array, tasks=task_config.tasks, reqs=req_tids_seq, cp_pops=cp_pops, cp_steps=cp_steps)
