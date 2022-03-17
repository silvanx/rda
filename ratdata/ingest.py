from dataclasses import dataclass
import numpy as np
import re
import h5py
import pathlib


@dataclass
class Recording:
    electrode_data: np.ndarray
    filename: str = ''
    dt: float = None
    time_of_recording: np.datetime_as_string = None
    rat_label: str = None
    recording_type: str = None
    stim_periods: tuple = ()


def extract_info_from_filename(filename: str) -> tuple[str, str, str]:
    date_regex = '([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}-[0-9]{2}-[0-9]{2}) '
    rat_regex = ' (rat[0-9]+) '

    try:
        time_of_recording = re.findall(date_regex, filename)[0]
    except IndexError:
        time_of_recording = None
    try:
        rat_label = re.findall(rat_regex, filename)[0]
    except IndexError:
        rat_label = None

    condition = None
    for condition_string in ['baseline', 'ST', 'OFT', 'CT']:
        c_rx = r'.* ' + re.escape(condition_string) + r'.*\.(mat|txt|bin|h5)'
        if re.match(c_rx, filename):
            condition = condition_string

    return(time_of_recording, rat_label, condition)


def extract_stim_type_from_filename(filename: str) -> str:
    random_regex = r'.* random\.mat$'
    on_off_regex = r'.* on-off.*\.mat$'
    continuous_regex = r'.* (130Hz|open-loop|DBS)\.mat$'
    proportional_regex = r'.* pro\.mat$'
    stim_type = None

    if re.match(random_regex, filename):
        stim_type = 'random'
    elif re.match(on_off_regex, filename):
        stim_type = 'on-off'
    elif re.match(continuous_regex, filename):
        stim_type = 'continuous'
    elif re.match(proportional_regex, filename):
        stim_type = 'proportional'
    return stim_type


def read_mce_matlab_file(filename: str) -> Recording:
    file = h5py.File(filename)
    prefix = '_'.join(list(file.keys())[0].split('_')[:-1])
    num_channels = len(list(file.keys()))
    time_of_recording, rat_label, recording_type = \
        extract_info_from_filename(filename)

    for i, channel_name in enumerate(['Ch2', 'Ch3', 'Ch4', 'Ch5']):
        key = '_'.join([prefix, channel_name])
        data = file.get(key)
        dt = data.get('interval')[0][0]
        samples = int(data.get('length')[0][0])

        raw_values = np.array(data.get('values')[0])
        if i == 0:
            electrode_data = np.zeros((4, samples))
        electrode_data[i, :] = raw_values

    stim_start_times = []
    stim_stop_times = []
    stim_periods = []
    for ch_number in range(1, num_channels):
        fieldname = ''.join([prefix, '_Ch', str(ch_number)])
        field_info = ''.join([chr(c[0]) for c in
                              file.get(fieldname).get('title')])
        if re.match('.* STG 1 Stimulation Start$', field_info) is not None:
            # print("%d: %s" % (channel_number, field_info))
            if file.get(fieldname).get('length')[0][0] > 0:
                stim_start_times = file.get(fieldname).get('times')[0]
        if re.match('.* STG 1 Stimulation Stop$', field_info) is not None:
            # print("%d: %s" % (channel_number, field_info))
            if file.get(fieldname).get('length')[0][0] > 0:
                stim_stop_times = file.get(fieldname).get('times')[0]

    if len(stim_start_times) > 0 and len(stim_stop_times) > 0:
        if stim_stop_times[0] < stim_start_times[0]:
            stim_stop_times = np.delete(stim_stop_times, 0)
        if stim_start_times[-1] > stim_stop_times[-1]:
            stim_start_times = np.delete(stim_start_times, -1)
        stim_periods = tuple([b for b in
                              zip(stim_start_times, stim_stop_times)])

    result = Recording(electrode_data, filename, dt, time_of_recording,
                       rat_label, recording_type, stim_periods)
    return result


def read_mce_hdf5_matlab_file(filename: str) -> Recording:
    raise NotImplementedError


def read_gui_csv_file(filename: str) -> np.ndarray:
    raise NotImplementedError


def read_gui_bin_file(filename: str, channels: int = 26) -> np.ndarray:
    """Read data from GUI binary file

    File is supposed to contain 32-bit integers from 4 electrode channels,
    sampled at the same rate as the MCE recording and 22 DSP channels.
    """
    raw_data = np.fromfile(filename, dtype=np.int32)
    data = np.reshape(raw_data, (-1, channels))
    return data


def read_gui_amplitude_file_data(filename: str) -> np.ndarray:
    """Read data from amplitude GUI recording file

    File is supposed to contain one integer per line, sampled at 100 Hz.
    Comments are indicated by --- at the beginning of the line.
    """
    with open(filename, 'r') as f:
        amplitude_data = [int(e) for e in f if not re.match("^--- .*", e)]
    if amplitude_data:
        return np.array(amplitude_data)
    else:
        raise ValueError


def replace_oob_samples(data: np.ndarray, cutoff: float = 1e6) -> np.ndarray:
    """Replace samples outside of bounds [-cutoff, cutoff]

    Parameters:

    data -- numpy array with the data in the (samples, channels) format

    cutoff -- the cutoff value for the replacement (default 1e6)

    """
    clean_data = np.copy(data)
    last_sample = clean_data.shape[0] - 1
    idx_sample, idx_channel = np.where(
        np.logical_or(clean_data > cutoff, clean_data < -cutoff))
    for (s, c) in zip(idx_sample, idx_channel):
        if s == 0:
            clean_data[s, c] = clean_data[s + 1, c]
        elif s == last_sample:
            clean_data[s, c] = clean_data[s - 1, c]
        else:
            clean_data[s, c] = (clean_data[s + 1, c] +
                                clean_data[s - 1, c]) / 2
    return clean_data


def read_stim_amplitude_from_gui_recording(file: pathlib.Path,
                                           max_current: float) -> np.ndarray:
    delta_dbs_ampl = max_current / 16
    amplitude_data = np.genfromtxt(file, dtype=np.int16, comments='---')
    current_amplitude = np.array([(i + 1) * delta_dbs_ampl if i > 0 else 0
                                  for i in amplitude_data])
    return current_amplitude
