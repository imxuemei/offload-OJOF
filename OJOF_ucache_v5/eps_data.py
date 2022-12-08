# -*- coding: utf-8 -*-
import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from OJOF_ucache_v5.env_config_v2 import make_dirs

CB91_Blue = '#2CBDFE'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'

rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 14
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
# plt.style.use('classic')
# cmap = plt.cm.get_cmap("Pastel2", 20)  # Paired
# mycolors = cmap(np.linspace(0, 1, 20))
mymarkers = ['+', 'x', '|', '.', '1', '2', '3', '4', '^', 'v', '>', '<']
# mycolors = ['red', 'green', 'darkorange', 'gold','deepskyblue', 'lightblue', 'darkseagreen','seagreen', 'coral', 'gray','lightgrey']
mycolors = [CB91_Blue, CB91_Violet, CB91_Amber, CB91_Pink, CB91_Green, CB91_Purple,  'darkseagreen', 'seagreen', 'coral',
            'gray', 'lightgrey']
linestyle_tuple = [
    ('solid', 'solid'),
    ('dashdot', 'dashdot'),
    ('dashed', 'dashed'),
    ('densely dashdotted', (0, (3, 1, 1, 1))),
    ('long dash with offset', (5, (10, 3))),
    ('dashdotted', (0, (3, 5, 1, 5))),
    ('dotted', 'dotted'),
    ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
    ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),
    # ('loosely dotted', (0, (1, 10))),
    ('loosely dashed', (0, (5, 10))),
    # ('densely dashed',        (0, (5, 1))),
    ('loosely dashdotted', (0, (3, 10, 1, 10))),
    ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
]

G_FIGSIZE = 3.5
G_DPI = 300
G_CI = 0.95
G_FORMAT = "eps"  # "eps" #"jpg"


def save_diff_bar_figure(settings, methods, method_values, xlabel, ylabel, figname, ncol=1, ylim_min=None,
                         figsize=None):
    make_dirs(f"{figname}.csv")
    data1d_to_csv(methods, method_values, f"{figname}.csv")
    with open(f"{figname}.csv", 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(settings)

    ylim_min = None if ylim_min is None else ylim_min
    figsize = (G_FIGSIZE, G_FIGSIZE) if figsize is None else figsize

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    bar_width = 0.5
    all_pos = np.arange(len(settings)) * 5.5
    num_methods = len(methods)
    for i in range(num_methods):
        i_pos = all_pos + (i - int(num_methods / 2)) * (bar_width + 0.1)
        ax.bar(i_pos, method_values[i], bar_width, label=methods[i], color=mycolors[i])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(all_pos)
    ax.set_xticklabels(settings)
    ax.set_ylim(bottom=ylim_min)
    ax.legend(ncol=ncol, fontsize=8)
    # ax.grid()
    fig.tight_layout()
    # plt.show()
    plt.savefig(f"{figname}.{G_FORMAT}", format=G_FORMAT, bbox_inches='tight', dpi=G_DPI)
    plt.close(fig)


def save_plot_figure(methods, method_values, xlabel, ylabel, figname, xticks=None, ncol=2, ylim_min=None,
                     fig_size=None):
    make_dirs(f"{figname}.csv")
    data1d_to_csv(methods, method_values, f"{figname}.csv")

    ylim_min = None if ylim_min is None else ylim_min
    fig_size = (G_FIGSIZE, G_FIGSIZE) if fig_size is None else fig_size

    fig = plt.figure(figsize=fig_size)
    for i in range(len(methods)):
        if xticks is None:
            plt.plot(method_values[i], label=methods[i], color=mycolors[i], linewidth=1,
                     linestyle=linestyle_tuple[i][1])  # , marker=mymarkers[i]))
        else:
            plt.plot(xticks, method_values[i], label=methods[i], color=mycolors[i], linewidth=1,
                     linestyle=linestyle_tuple[i][1])  # , marker=mymarkers[i])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xticks is None:
        plt.xticks(np.arange(0, len(method_values[0]) + 1, step=int(len(method_values[0]) / 5)))
    plt.ylim(bottom=ylim_min)
    plt.legend(ncol=ncol, fontsize=8)
    # plt.grid()
    plt.tight_layout()
    plt.savefig(f"{figname}.{G_FORMAT}", format=G_FORMAT, bbox_inches='tight', dpi=G_DPI)
    plt.close(fig)


def data1d_to_csv(names, values, csv_name):
    data = np.column_stack((names, values))
    data = np.transpose(data)
    with open(csv_name, 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)

def read_csv_data1d(csv_name):
    data = []
    with open(csv_name, 'r') as f:
        for row in csv.reader(f):
            data.append(row)
    methods = data[0]
    values = np.array(data[1:-1], dtype=np.float)
    settings = data[-1]
    return methods, values, settings
