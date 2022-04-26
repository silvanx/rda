import datetime
import matplotlib.pyplot as plt
from ratdata import data_manager as dm, process, ingest
import numpy as np
import pandas as pd
import seaborn as sns


def plot_beta_one_rat_one_condition(rat_full_label: str, cond: str,
                                    img_filename: str = None,
                                    remove_oof: bool = False) -> None:
    rat = dm.Rat().get(full_label=rat_full_label)
    stim_array = ['nostim', 'continuous', 'on-off', 'random']
    boxplot_data = []
    for stim in stim_array:
        rec_array = dm.select_recordings_for_rat(rat, cond, stim)
        if remove_oof:
            beta = []
            for f in rec_array:
                if not dm.is_recording_rejected(f.filename):
                    m = f.power.get().oof_exponent
                    b = f.power.get().oof_constant
                    oof = process.oof_power_in_frequency_band(m, b, 12, 18)
                    power = f.power.get().beta_power - oof
                    beta.append(power)
            plot_title = 'Absolute beta power %s %s (without 1/f component)'\
                % (rat_full_label, cond)
        else:
            beta = [f.power.get().beta_power for f in rec_array
                    if not dm.is_recording_rejected(f.filename)]
            plot_title = 'Absolute beta power %s %s' % (rat_full_label, cond)
        boxplot_data.append(beta)
    boxplot_all_stim(boxplot_data, stim_array, plot_title, img_filename)


def plot_beta_one_rat(rat_full_label: str, img_filename: str = None,
                      remove_oof: bool = False) -> None:
    label_order = ['nostim', 'continuous', 'on-off', 'random']
    rat = dm.Rat().get(full_label=rat_full_label)
    rec_array = dm.RecordingFile.select().where(dm.RecordingFile.rat == rat)
    if rec_array.count() == 0:
        return
    plot_data = []
    for f in rec_array:
        if dm.is_recording_rejected(f.filename):
            continue
        if remove_oof:
            m = f.power.get().oof_exponent
            b = f.power.get().oof_constant
            oof = process.oof_power_in_frequency_band(m, b, 12, 18)
            power = f.power.get().beta_power - oof
            plot_title = 'Absolute beta power %s (without 1/f component)'\
                % (rat_full_label)
        else:
            power = f.power.get().beta_power
            plot_title = 'Absolute beta power %s' % (rat_full_label)
        if f.stim.count() == 0:
            stim = 'nostim'
        else:
            stim = f.stim.get().stim_type
        plot_data.append([power, stim])
    df = pd.DataFrame(plot_data)
    df.columns = ['power', 'stim']
    fig = plt.figure(figsize=(12, 6))
    sns.boxplot(x='stim', y='power', data=df, palette='Dark2',
                order=label_order, boxprops=dict(alpha=.8))
    sns.swarmplot(x='stim', y='power', data=df, s=4, palette='Dark2',
                  order=label_order)
    plt.title(plot_title)
    save_or_show(fig, img_filename)


def plot_relative_beta_one_rat_one_condition(rat_full_label: str,
                                             cond: str,
                                             img_filename: str = None) -> None:
    rat = dm.Rat().get(full_label=rat_full_label)
    stim_array = ['nostim', 'continuous', 'on-off', 'random']
    boxplot_data = []
    for stim in stim_array:
        rec_array = dm.select_recordings_for_rat(rat, cond, stim)
        rbeta = [f.power.get().beta_power / f.power.get().total_power
                 for f in rec_array
                 if not dm.is_recording_rejected(f.filename)]
        boxplot_data.append(rbeta)
    plot_title = 'Relative beta power %s %s' % (rat_full_label, cond)
    boxplot_all_stim(boxplot_data, stim_array, plot_title, img_filename)


def plot_change_in_absolute_beta(rat_full_label: str,
                                 cond: str,
                                 img_filename: str = None,
                                 remove_oof: bool = False) -> None:
    rat = dm.Rat().get(full_label=rat_full_label)
    stim_array = ['nostim', 'continuous', 'on-off', 'random']
    boxplot_data = []
    plot_title = 'Change in absolute beta power %s %s' % (rat_full_label, cond)
    for stim in stim_array:
        rec_array = dm.select_recordings_for_rat(rat, cond, stim)
        beta_change = [process.get_change_in_beta_power_from_rec(f,
                                                                 remove_oof)
                       for f in rec_array
                       if not dm.is_recording_rejected(f.filename)
                       and f.baseline.count() > 0]
        boxplot_data.append(beta_change)
    boxplot_all_stim(boxplot_data, stim_array, plot_title, img_filename)


def plot_change_in_relative_beta(rat_full_label: str,
                                 cond: str,
                                 img_filename: str = None) -> None:
    rat = dm.Rat().get(full_label=rat_full_label)
    stim_array = ['nostim', 'continuous', 'on-off', 'random']
    boxplot_data = []
    plot_title = 'Change in relative beta power %s %s' % (rat_full_label, cond)
    for stim in stim_array:
        rec_array = dm.select_recordings_for_rat(rat, cond, stim)
        rbeta_change = [process.get_change_in_rel_beta_power_from_rec(f)
                        for f in rec_array
                        if not dm.is_recording_rejected(f.filename)]
        rbeta_change = [e for e in rbeta_change if e is not None]
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


def plot_baseline_across_time(rat_full_label: str,
                              img_filename: str = None) -> None:
    time_slices_file = 'data/mce_recordings/time_slices.pickle'
    # Read time slices from the file
    time_slices = ingest.read_file_slices(time_slices_file)
    rat = dm.Rat.get(full_label=rat_full_label)
    baseline_recordings = dm.RecordingFile.select()\
        .where((dm.RecordingFile.rat == rat) &
               (dm.RecordingFile.condition == 'baseline'))\
        .order_by(dm.RecordingFile.recording_date)
    plot_power = []
    plot_date = []
    for rec in baseline_recordings:
        if not file_rejected(time_slices, rec.filename):
            power_data = dm.RecordingPower.get(recording=rec)
            relative_power = power_data.beta_power / power_data.total_power
            plot_power.append(relative_power)
            plot_date.append(rec.recording_date)

    fig = plt.figure(figsize=(12, 6))
    plt.plot(plot_date, plot_power, '.-')
    plt.title('Baseline relative beta for %s' % rat_full_label)

    save_or_show(fig, img_filename)


def plot_relative_beta_one_day(day: datetime.date,
                               filename_prefix: str = None,
                               ignore_stim: bool = False) -> None:
    rats = dm.RecordingFile.select().join(dm.Rat)\
        .where(dm.RecordingFile.recording_date == day)\
        .group_by(dm.RecordingFile.rat)
    for record in rats:
        rat = record.rat
        plot_relative_beta_one_day_one_rat(day, rat, filename_prefix,
                                           ignore_stim)


def file_rejected(time_slices, filename):
    file_key = filename.split('.')[0]
    if (file_key in time_slices and
            'reject' in time_slices[file_key] and
            time_slices[file_key]['reject']):
        return True
    else:
        return False


def plot_relative_beta_one_day_one_rat(day: datetime.date,
                                       rat: dm.Rat,
                                       filename_prefix: str = None,
                                       ignore_stim: bool = False) -> None:
    time_slices_file = 'data/mce_recordings/time_slices.pickle'
    # Read time slices from the file
    time_slices = ingest.read_file_slices(time_slices_file)
    if ignore_stim:
        recordings = dm.RecordingFile.select()\
            .join(dm.StimSettings)\
            .where((dm.RecordingFile.recording_date == day) &
                   (dm.RecordingFile.rat == rat) &
                   (dm.StimSettings.stim_type == 'nostim'))\
            .order_by(dm.RecordingFile.filename)
    else:
        recordings = dm.RecordingFile.select()\
            .where((dm.RecordingFile.recording_date == day) &
                   (dm.RecordingFile.rat == rat))\
            .order_by(dm.RecordingFile.filename)
    if recordings.count() > 0:
        recordings = dm.RecordingFile.select()\
            .where((dm.RecordingFile.recording_date == day) &
                   (dm.RecordingFile.rat == rat))\
            .order_by(dm.RecordingFile.filename)
        data_list = [(r.condition,
                      dm.RecordingPower.get(recording=r).beta_power)
                     for r in recordings
                     if not file_rejected(time_slices, r.filename)]
        if data_list:
            label, rbeta = list(zip(*data_list))
            if rbeta:
                fig = plt.figure(figsize=(12, 6))
                ax = plt.gca()
                plt.bar(range(len(rbeta)), rbeta)
                ax.set_xticks(range(len(rbeta)))
                ax.set_xticklabels(label)
                plt.title('Relative beta power for %s on %s' %
                          (rat.full_label, day.strftime("%d %b %Y")))
                if filename_prefix is not None:
                    filename = '%s_%s_%s.png' % (filename_prefix,
                                                 rat.full_label,
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


def plot_biomarker_steps(data: np.ndarray, fs: int,
                         time: tuple[float, float] = None, low_fs: int = 500,
                         lowcut: int = 13, hicut: int = 30,
                         p_seg_len: int = 50, plot_title: str = None,
                         filename: str = None) -> None:
    if time is None:
        start_time = 0
        maxtime = len(data) / fs
    else:
        start_time = time[0]
        maxtime = time[1]
    downsampled = process.downsample_signal(data, fs, low_fs)
    beta = process.bandpass_filter(downsampled, low_fs, lowcut, hicut)
    beta_no_mean = process.bandpass_filter(downsampled - np.mean(downsampled),
                                           low_fs, lowcut, hicut)
    beta_power = process.rolling_power_signal(beta, p_seg_len)
    rms_beta = np.sqrt(np.mean(beta ** 2))
    rms_beta_no_mean = np.sqrt(np.mean((beta_no_mean) ** 2))
    print('RMS beta: %f, RMS beta without mean: %f' %
          (rms_beta, rms_beta_no_mean))
    total_power = process.rolling_power_signal(downsampled, p_seg_len)
    biomarker = beta_power / total_power
    p_cut = int(p_seg_len / 2)
    tt = np.linspace(start_time, maxtime, len(data))
    ttd = np.linspace(start_time, maxtime, len(downsampled))

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


def plot_amplitude_with_percentiles(plot_data: np.ndarray, tstart: float = 0,
                                    tstop: float = None,
                                    fs: int = 200) -> None:
    p10 = np.percentile(plot_data, 10)
    p20 = np.percentile(plot_data, 20)
    plt.figure(figsize=(20, 10), dpi=100)
    if tstop is None or tstop * fs > len(plot_data):
        tstop = int(len(plot_data) / fs)
    ttn = (tstop - tstart) * fs
    plt.plot(np.linspace(tstart, tstop, ttn),
             plot_data[tstart * fs:tstop * fs])
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [uA]')
    plt.axhline(np.mean(plot_data), linestyle='-', color='k')
    plt.axhline(p20, linestyle='--', color='k')
    plt.axhline(p10, linestyle=':', color='k')
    plt.legend(['stim amplitude [uA]', 'mean amplitude',
                '20th percentile', '10th percentile'])


def plot_peak_location_and_height(peaks: dict, title_remark: str,
                                  plot_filename: str = None,
                                  x_lim: list[float] = [11, 25],
                                  y_lim: list[float] = [-3e-5, 1.5e-4])\
                                      -> None:
    c = plt.get_cmap('Dark2')
    control_rats = [r.label for r in
                    dm.Rat.select().where(dm.Rat.group == 'control')]
    ohda_rats = [r.label for r in
                 dm.Rat.select().where(dm.Rat.group == '6OHDA')]
    columns = 2
    rows = max([len(control_rats), len(ohda_rats)])

    fig, ax = plt.subplots(nrows=rows, ncols=columns, figsize=(12, 16))
    for i, rat in enumerate(ohda_rats):
        ax[i, 0].set_title(rat)
        ax[i, 0].set_xlim(x_lim)
        ax[i, 0].set_ylim(y_lim)
        xx, yy, pp = zip(*peaks[rat])
        ax[i, 0].scatter(xx, yy, s=10, color=c.colors[i])
        ax[i, 0].axvline(np.mean(xx), color=c.colors[i], alpha=1)
        ax[i, 0].fill_betweenx(ax[i, 0].get_ylim(),
                               np.ones(2) * (np.mean(xx) - np.std(xx)),
                               np.ones(2) * (np.mean(xx) + np.std(xx)),
                               color=c.colors[i], alpha=0.2)

    for i, rat in enumerate(control_rats):
        ax[i, 1].set_title(rat)
        ax[i, 1].set_xlim(x_lim)
        ax[i, 1].set_ylim(y_lim)
        xx, yy, pp = zip(*peaks[rat])
        ax[i, 1].scatter(xx, yy, s=10, color=c.colors[len(ohda_rats) + i])
        ax[i, 1].axvline(np.mean(xx), color=c.colors[len(ohda_rats) + i],
                         alpha=1)
        ax[i, 1].fill_betweenx(ax[i, 1].get_ylim(),
                               np.ones(2) * (np.mean(xx) - np.std(xx)),
                               np.ones(2) * (np.mean(xx) + np.std(xx)),
                               color=c.colors[len(ohda_rats) + i], alpha=0.2)

    ax[0, 0].annotate('6-OHDA rats', xy=(0, 0), va='center', ha='center',
                      xycoords=ax[0, 0].title, xytext=(0.5, 2.2), fontsize=14)
    ax[0, 1].annotate('control rats', xy=(0, 0), va='center', ha='center',
                      xycoords=ax[0, 1].title, xytext=(0.5, 2.2), fontsize=14)
    ax[3, 1].axis('off')
    fig.suptitle('Most prominent peak location and height%s' % title_remark,
                 fontsize=14)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.23, top=0.92)

    save_or_show(fig, plot_filename)
