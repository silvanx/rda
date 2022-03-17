import matplotlib.pyplot as plt
from ratdata import data_manager as dm, process
import numpy as np


def plot_beta_one_rat_one_condition(rat_label: str, cond: str,
                                    img_filename: str = None) -> None:
    rat = dm.Rat().get(label=rat_label)
    stim_array = ['continuous', 'on-off', 'random']
    boxplot_data = []
    for stim in stim_array:
        rec_array = dm.select_recordings_for_rat(rat, cond, stim)
        beta = [f.power.get().beta_power for f in rec_array]
        boxplot_data.append(beta)
    plot_title = 'Absolute beta power %s %s' % (rat_label, cond)
    boxplot_all_stim(boxplot_data, stim_array, plot_title, img_filename)


def plot_relative_beta_one_rat_one_condition(rat_label: str,
                                             cond: str,
                                             img_filename: str = None) -> None:
    rat = dm.Rat().get(label=rat_label)
    stim_array = ['continuous', 'on-off', 'random']
    boxplot_data = []
    for stim in stim_array:
        rec_array = dm.select_recordings_for_rat(rat, cond, stim)
        rbeta = [f.power.get().beta_power / f.power.get().total_power
                 for f in rec_array]
        boxplot_data.append(rbeta)
    plot_title = 'Relative beta power %s %s' % (rat_label, cond)
    boxplot_all_stim(boxplot_data, stim_array, plot_title, img_filename)


def plot_change_in_absolute_beta(rat_label: str,
                                 cond: str,
                                 img_filename: str = None) -> None:
    rat = dm.Rat().get(label=rat_label)
    stim_array = ['continuous', 'on-off', 'random']
    boxplot_data = []
    plot_title = 'Change in absolute beta power %s %s' % (rat_label, cond)
    for stim in stim_array:
        rec_array = dm.select_recordings_for_rat(rat, cond, stim)
        rbeta_change = [process.get_change_in_beta_power_from_rec(f)
                        for f in rec_array]
        boxplot_data.append(rbeta_change)
    boxplot_all_stim(boxplot_data, stim_array, plot_title, img_filename)


def plot_change_in_relative_beta(rat_label: str,
                                 cond: str,
                                 img_filename: str = None) -> None:
    rat = dm.Rat().get(label=rat_label)
    stim_array = ['continuous', 'on-off', 'random']
    boxplot_data = []
    plot_title = 'Change in relative beta power %s %s' % (rat_label, cond)
    for stim in stim_array:
        rec_array = dm.select_recordings_for_rat(rat, cond, stim)
        rbeta_change = [process.get_change_in_rel_beta_power_from_rec(f)
                        for f in rec_array]
        boxplot_data.append(rbeta_change)
    boxplot_all_stim(boxplot_data, stim_array, plot_title, img_filename)


def boxplot_all_stim(boxplot_data: list[list[float]], x_labels: list[str],
                     title: str = '', img_filename: str = None) -> None:
    plt.figure(figsize=(12, 6))
    plt.boxplot(boxplot_data)
    for i, data_points in enumerate(boxplot_data):
        plt.scatter(np.ones(len(data_points)) * (i + 1), data_points)
    plt.title(title)
    ax = plt.gca()
    ax.set_xticklabels(x_labels)

    if img_filename:
        plt.savefig(img_filename, facecolor='white', bbox_inches='tight')
        plt.close()
    else:
        plt.show()
