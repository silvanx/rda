import datetime
import matplotlib.pyplot as plt
from ratdata import data_manager as dm, process, ingest
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
    time_slices_file = 'data/mce_recordings/time_slices.pickle'
    # Read time slices from the file
    time_slices = ingest.read_file_slices(time_slices_file)
    rat = dm.Rat().get(label=rat_label)
    stim_array = ['nostim', 'continuous', 'on-off', 'random']
    boxplot_data = []
    for stim in stim_array:
        rec_array = dm.select_recordings_for_rat(rat, cond, stim)
        rbeta = [f.power.get().beta_power / f.power.get().total_power
                 for f in rec_array
                 if not file_rejected(time_slices, f.filename)]
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
    time_slices_file = 'data/mce_recordings/time_slices.pickle'
    # Read time slices from the file
    time_slices = ingest.read_file_slices(time_slices_file)
    rat = dm.Rat().get(label=rat_label)
    stim_array = ['nostim', 'continuous', 'on-off', 'random']
    boxplot_data = []
    plot_title = 'Change in relative beta power %s %s' % (rat_label, cond)
    for stim in stim_array:
        rec_array = dm.select_recordings_for_rat(rat, cond, stim)
        rbeta_change = [process.get_change_in_rel_beta_power_from_rec(f)
                        for f in rec_array
                        if not file_rejected(time_slices, f.filename)]
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


def plot_baseline_across_time(rat_label: str,
                              img_filename: str = None) -> None:
    time_slices_file = 'data/mce_recordings/time_slices.pickle'
    # Read time slices from the file
    time_slices = ingest.read_file_slices(time_slices_file)
    rat = dm.Rat.get(label=rat_label)
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
    plt.title('Baseline relative beta for %s' % rat_label)

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


def plot_amplitude_with_percentiles(plot_data, tstart=0, tstop=None, fs=200):
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
    plt.legend(['rat2 stim amplitude [uA]', 'mean amplitude',
                '20th percentile', '10th percentile'])
