# coding=utf-8
MAB_I_LINUCB = "iLinUCB"
MAB_LINUCB = "LinUCB"
MAB_UCB1 = "UCB1"
MAB_E_GREEDY = r"$\epsilon$-greedy"
# MAB_PERFECT = "Perfect"

OFFLOAD_LOCAL = "Local"
OFFLOAD_REMOTE = "Remote"
OFFLOAD_RANDOM = "Random"
OFFLOAD_GREEDY = "Greedy"

ARM_C_L = 1
ARM_C_R = 2
ARM_NOC_L = 3
ARM_NOC_R = 4
ARM_NOC_R2 = 5
ARM_LIST = [1, 2, 3, 4]
NO_CACHE = 0
CACHED = 1
NO_TASK = -1

CACHE_SW_SIZE = 300
CA_LFU = "LFU"  # long term frequency
CA_W_LFU = "W-LFU"  # sliding window frequency
CA_TLCPD = "TLCPD-based OCA"
CA_PERFECT = "Perfect"
CA_NO_CACHE = "NoCache"
CA_PHT = "PHT"
CA_ADWIN = "ADWIN"

DEFAULT_CA_PARAMS = {
    CA_LFU: {"ca_type": CA_LFU, },
    CA_W_LFU: {"ca_type": CA_W_LFU, },
    CA_TLCPD: {
            "ca_type": CA_TLCPD,
            "cpd_topK": 3,
            "cpd_config": {
                "cp_cluster_ws": 20,
                "ft_pht_threshold": 0.5,
                "ft_pht_tv_num": 70,
                "kl_pht_threshold": 0.1,
                "kl_pht_tv_num": 50,},
        },
    CA_PHT: {
        "ca_type": CA_PHT,
        "cpd_topK": 3,
        "cpd_config": {"threshold": 1.0, },
    },
    CA_ADWIN: {
        "ca_type": CA_ADWIN,
        "cpd_topN": 3,
        "cpd_config": {"delta": 0.9999, },
    },
}

DEFAULT_BANDIT_PARAMS = {
    MAB_I_LINUCB: {
        "alpha": 0.01,
        "df_reward": 0.01,
    },
    # MAB_PERFECT: None,
    MAB_LINUCB: {
        "alpha": 0.01,
    },
    MAB_E_GREEDY: {"epsilon": 0.01, },
    MAB_UCB1: None,
}
