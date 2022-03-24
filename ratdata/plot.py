import datetime
import matplotlib.pyplot as plt
from ratdata import data_manager as dm, process
import numpy as np


def plot_beta_one_rat_one_condition(rat_label: str, cond: str,
                                    img_filename: str = None) -> None:
    rat = dm.Rat().get(label=rat_label)
    stim_array = ['nostim', 'continuous', 'on-off', 'random']
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
    stim_array = ['nostim', 'continuous', 'on-off', 'random']
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
    stim_array = ['nostim', 'continuous', 'on-off', 'random']
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
    stim_array = ['nostim', 'continuous', 'on-off', 'random']
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
    fig = plt.figure(figsize=(12, 6))
    plt.boxplot(boxplot_data)
    for i, data_points in enumerate(boxplot_data):
        plt.scatter(np.ones(len(data_points)) * (i + 1), data_points)
    plt.title(title)
    ax = plt.gca()
    ax.set_xticklabels(x_labels)

    save_or_show(fig, img_filename)


def plot_baseline_across_time(rat_label: str,
                              img_filename: str = None) -> None:
    rat = dm.Rat.get(label=rat_label)
    baseline_recordings = dm.RecordingFile.select()\
        .where((dm.RecordingFile.rat == rat) &
               (dm.RecordingFile.condition == 'baseline'))\
        .order_by(dm.RecordingFile.recording_date)
    plot_power = []
    plot_date = []
    for rec in baseline_recordings:
        power_data = dm.RecordingPower.get(recording=rec)
        relative_power = power_data.beta_power / power_data.total_power
        plot_power.append(relative_power)
        plot_date.append(rec.recording_date)

    fig = plt.figure(figsize=(12, 6))
    plt.plot(plot_date, plot_power, '.-')
    plt.title('Baseline relative beta for %s' % rat_label)

    save_or_show(fig, img_filename)


def plot_relative_beta_one_day(day: datetime.date,
                               filename_prefix: str = None) -> None:
    rats = dm.RecordingFile.select().join(dm.Rat)\
        .where(dm.RecordingFile.recording_date == day)\
        .group_by(dm.RecordingFile.rat)
    for record in rats:
        rat = record.rat
        plot_relative_beta_one_day_one_rat(day, rat, filename_prefix)


def plot_relative_beta_one_day_one_rat(day: datetime.date,
                                       rat: dm.Rat,
                                       filename_prefix: str = None) -> None:

    recordings = dm.RecordingFile.select()\
        .where((dm.RecordingFile.recording_date == day) &
               (dm.RecordingFile.rat == rat))\
        .order_by(dm.RecordingFile.filename)
    data_list = [(r.condition,
                  dm.RecordingPower.get(recording=r).beta_power)
                 for r in recordings]
    label, rbeta = list(zip(*data_list))
    fig = plt.figure(figsize=(12, 6))
    plt.bar(label, rbeta)
    plt.title('Relative beta power for %s on %s' %
              (rat.label, day.strftime("%d %b %Y")))
    if filename_prefix is not None:
        filename = '%s_%s_%s.png' % (filename_prefix, rat.label,
                                     day.strftime("%Y%m%d"))
    else:
        filename = None
    save_or_show(fig, filename)


def save_or_show(fig: plt.Figure, filename: str = None) -> None:
    plt.figure(fig)
    if filename is not None:
        plt.savefig(filename, facecolor='white', bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_biomarker_steps(data: np.ndarray, fs: int, low_fs: int = 500,
                         lowcut: int = 13, hicut: int = 30,
                         p_seg_len: int = 50, plot_title: str = None,
                         filename: str = None) -> None:
    maxtime = len(data) / fs
    downsampled = process.downsample_signal(data, fs, low_fs)
    beta = process.bandpass_filter(downsampled, low_fs, lowcut, hicut)
    beta_power = process.rolling_power_signal(beta, p_seg_len)
    total_power = process.rolling_power_signal(downsampled, p_seg_len)
    biomarker = beta_power / total_power
    p_cut = int(p_seg_len / 2)
    tt = np.linspace(0, maxtime, len(data))
    ttd = np.linspace(0, maxtime, len(downsampled))

    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(14, 10))
    ax[0].plot(tt, data)
    ax[0].set_title('Raw data')
    ax[1].plot(ttd, beta)
    ax[1].set_title('Beta (13-30 Hz) component')
    ax[2].plot(ttd[p_cut:-p_cut], beta_power[p_cut:-p_cut])
    ax[2].plot(ttd[p_cut:-p_cut], total_power[p_cut:-p_cut])
    ax[2].legend(['beta', 'total'])
    ax[2].set_title('Beta and total power (N = %d samples)' % p_seg_len)
    ax[3].plot(ttd[p_cut:-p_cut], biomarker[p_cut:-p_cut])
    ax[3].set_title('Biomarker (relative beta)')
    if plot_title is not None:
        plt.suptitle(plot_title)

    ax[-1].set_xlabel('Time [s]')

    save_or_show(fig, filename)
