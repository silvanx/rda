import matplotlib.pyplot as plt
from ratdata import data_manager as dm
import numpy as np


def plot_beta_one_rat_one_condition(rat_label: str, cond: str) -> None:
    example_rat = dm.Rat().get(label=rat_label)
    stim_array = ['continuous', 'on-off', 'random']

    plt.figure(figsize=(12, 6))
    boxplot_data = []
    for stim in stim_array:
        rec_array = dm.select_recordings_for_rat(example_rat, cond, stim)
        beta = [f.power.get().beta_power for f in rec_array]
        boxplot_data.append(beta)
    plt.boxplot(boxplot_data)
    for i, data_points in enumerate(boxplot_data):
        plt.scatter(np.ones(len(data_points)) * (i + 1), data_points)
    ax = plt.gca()
    ax.set_xticklabels(stim_array)
    plt.title('Absolute beta power %s %s' % (rat_label, cond))
    plt.show()


def plot_relative_beta_one_rat_one_condition(rat_label: str,
                                             cond: str) -> None:
    example_rat = dm.Rat().get(label=rat_label)
    stim_array = ['continuous', 'on-off', 'random']

    plt.figure(figsize=(12, 6))
    boxplot_data = []
    for stim in stim_array:
        rec_array = dm.select_recordings_for_rat(example_rat, cond, stim)
        rbeta = [f.power.get().beta_power / f.power.get().total_power
                 for f in rec_array]
        boxplot_data.append(rbeta)
    plt.boxplot(boxplot_data)
    for i, data_points in enumerate(boxplot_data):
        plt.scatter(np.ones(len(data_points)) * (i + 1), data_points)
    ax = plt.gca()
    ax.set_xticklabels(stim_array)
    plt.title('Relative beta power %s %s' % (rat_label, cond))
    plt.show()
