from dataclasses import dataclass
from collections.abc import Sequence
import numpy as np
import re
import h5py
import pathlib
import pickle


@dataclass
class Recording:
    electrode_data: np.ndarray
    filename: str = ''
    dt: float = None
    time_of_recording: np.datetime_as_string = None
    rat_label: str = None
    recording_type: str = None
    stim_periods: tuple = ()
    pulse_periods: tuple = ()
    _slice: tuple[float, float] = None

    @property
    def slice(self) -> tuple[float, float]:
        return self._slice

    @slice.setter
    def slice(self, s: tuple[float, float]) -> None:
        self._slice = s


def extract_info_from_filename(filename: str) -> tuple[str, str, str]:
    date_regex = r'([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}-[0-9]{2}-[0-9]{2}) '
    rat_regex = r' (rat[0-9]+) '

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
    continuous_regex = r'.* (130Hz|open-loop|DBS|no)\.mat$'
    proportional_regex = r'.* pro\.mat$'
    low_regex = r'.* low\.mat$'
    low20_regex = r'.* low20\.mat$'
    stim_type = 'nostim'

    if re.match(random_regex, filename):
        stim_type = 'random'
    elif re.match(on_off_regex, filename):
        stim_type = 'on-off'
    elif re.match(continuous_regex, filename):
        stim_type = 'continuous'
    elif re.match(proportional_regex, filename):
        stim_type = 'proportional'
    elif re.match(low_regex, filename):
        stim_type = 'low'
    elif re.match(low20_regex, filename):
        stim_type = 'low20'
    return stim_type


def read_file_slices(filename: str) -> dict:
    file = pathlib.Path(filename)
    if file.exists():
        with open(file, 'rb') as f:
            time_slices = pickle.load(f)
    else:
        time_slices = dict()

    return time_slices


def export_stim_events(file: h5py.File, fieldname: str) -> list[float]:
    result = []
    if fieldname is not None:
        if file.get(fieldname).get('length')[0][0] > 0:
            result = file.get(fieldname).get('times')[0]
    return result


def periods_from_start_stop(start: list[float],
                            stop: list[float]) -> Sequence[tuple[float,
                                                                 float]]:
    periods = []
    if len(start) > 0 and len(stop) > 0:
        if stop[0] < start[0]:
            stop = np.delete(stop, 0)
        if len(stop) == 0 or start[-1] > stop[-1]:
            start = np.delete(start, -1)
        periods = tuple([b for b in
                        zip(start, stop)])
    return periods


def read_mce_matlab_file(filename: str) -> Recording:
    filename = str(filename)
    file = h5py.File(filename)
    prefix_list = list(file.keys())[0].split('_')
    prefix_list[-1] = 'Ch'
    prefix = '_'.join(prefix_list)
    num_channels = len(list(file.keys()))
    time_of_recording, rat_label, recording_type = \
        extract_info_from_filename(filename)

    field_names = {
        'E:E1': {
            'regex': '^E:E1$',
            'number': None,
            'name': None
        },
        'E:E2': {
            'regex': '^E:E2$',
            'number': None,
            'name': None
        },
        'E:E3': {
            'regex': '^E:E3$',
            'number': None,
            'name': None
        },
        'E:E4': {
            'regex': '^E:E4$',
            'number': None,
            'name': None
        },
        'STG 1 Start': {
            'regex': '.* STG 1 Stimulation Start$',
            'number': None,
            'name': None
        },
        'STG 1 Stop': {
            'regex': '.* STG 1 Stimulation Stop$',
            'number': None,
            'name': None
        },
        'Pulse Start': {
            'regex': '.* STG 1 Single Pulse Start$',
            'number': None,
            'name': None
        },
        'Pulse Stop': {
            'regex': '.* STG 1 Single Pulse Stop$',
            'number': None,
            'name': None
        },
    }

    for ch_number in range(1, num_channels):
        fieldname = ''.join([prefix, str(ch_number)])
        field_info = ''.join([chr(c[0]) for c in
                              file.get(fieldname).get('title')])
        for k, v in field_names.items():
            if re.match(v['regex'], field_info):
                old_number = field_names[k]['number']
                if old_number is None or ch_number < old_number:
                    field_names[k]['number'] = ch_number
                    field_names[k]['name'] = fieldname

    for i, channel_name in enumerate(['E:E1', 'E:E2', 'E:E3', 'E:E4']):
        key = field_names[channel_name]['name']
        data = file.get(key)
        dt = data.get('interval')[0][0]
        samples = int(data.get('length')[0][0])

        raw_values = np.array(data.get('values')[0])
        if i == 0:
            electrode_data = np.zeros((4, samples))
        electrode_data[i, :] = raw_values

    stim_start = export_stim_events(file,
                                    field_names['STG 1 Start']['name'])

    stim_stop = export_stim_events(file,
                                   field_names['STG 1 Stop']['name'])
    pulse_start = export_stim_events(file,
                                     field_names['Pulse Start']['name'])
    pulse_stop = export_stim_events(file,
                                    field_names['Pulse Stop']['name'])

    stim_periods = periods_from_start_stop(stim_start, stim_stop)

    pulse_periods = periods_from_start_stop(pulse_start, pulse_stop)

    result = Recording(electrode_data, filename, dt, time_of_recording,
                       rat_label, recording_type, stim_periods, pulse_periods)
    return result


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

    File is supposed to contain one integer per line, sampled at 200 Hz.
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
