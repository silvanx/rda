import scipy.signal as signal
import scipy.integrate as integrate
import numpy as np
from ratdata import data_manager as dm, ingest


def compute_power_in_frequency_band(data: np.ndarray, low: int, high: int,
                                    fs: int) -> float:
    freqs, psd = signal.welch(data, fs, nperseg=fs)
    idx = np.logical_and(freqs >= low, freqs <= high)
    freqs_res = freqs[1] - freqs[0]
    power = integrate.trapz(psd[idx], dx=freqs_res)
    return power


def downsample_signal(data: np.ndarray, fs: int, target_fs: int) -> np.ndarray:
    q = int(fs / target_fs)
    decimated = signal.decimate(data, q, ftype='fir')

    return decimated


def bandpass_filter(data: np.ndarray, fs: int, lowcut: int,
                    hicut: int, numtaps: int = 201) -> np.ndarray:
    nyq = fs / 2
    lo = lowcut / nyq
    hi = hicut / nyq
    taps = signal.firwin(numtaps, (lo, hi), pass_zero='bandpass')
    filtered = signal.lfilter(taps, 1.0, data)
    return filtered


def compute_relative_power(data: np.ndarray, low: int, high: int,
                           fs: int, total_low: int = 0,
                           total_high: int = None) -> float:
    if not total_high:
        total_high = fs / 2
    band_power = compute_power_in_frequency_band(data, low, high, fs)
    total_power = compute_power_in_frequency_band(data, total_low, total_high,
                                                  fs)
    relative_power = band_power / total_power
    return relative_power


def rolling_power_signal(data: np.ndarray, segment_len: int) -> np.ndarray:
    d2 = data ** 2
    power = np.convolve(d2, np.ones(segment_len) / segment_len, mode='same')
    return power


def compute_change_in_power(data1: np.ndarray, data2: np.ndarray, low: int,
                            high: int, fs: int) -> float:
    power1 = compute_power_in_frequency_band(data1, low, high, fs)
    power2 = compute_power_in_frequency_band(data2, low, high, fs)
    return (power2 / power1 - 1) * 100


def get_change_in_beta_power_from_rec(rec: dm.RecordingFile) -> float:
    baseline = rec.baseline.get().baseline
    recording_power = dm.RecordingPower.get(recording=rec)
    baseline_power = dm.RecordingPower.get(recording=baseline)
    power_change = (recording_power.beta_power /
                    baseline_power.beta_power - 1) * 100
    return power_change


def get_change_in_rel_beta_power_from_rec(rec: dm.RecordingFile) -> float:
    if rec.baseline.count() == 1:
        baseline = rec.baseline.get().baseline
        recording_power = dm.RecordingPower.get(recording=rec)
        baseline_power = dm.RecordingPower.get(recording=baseline)
        recording_rel_power = (recording_power.beta_power /
                               recording_power.total_power)
        baseline_rel_power = (baseline_power.beta_power /
                              baseline_power.total_power)
        power_change = (recording_rel_power /
                        baseline_rel_power - 1) * 100
        return power_change


def compute_teed_continuous_stim(amplitude: int, pulse_width: int,
                                 stim_frequency: float,
                                 impedance: float) -> float:
    '''
    Compute TEED using the formula in (Helmers, 2017)

    TEED is calculated using the formula (V^2 * f * pw) / R * (1 sec)

    Since we know the stimulation current, not the stimulation voltage, we get
    (I^2 * f * pw * R) * (1 sec)


    Parameters
    ----------
    amplitude : int
        Stimulation amplitude in uA
    pulse_width : int
        Pulse width in us
    stim_frequency : float
        Stimulation frequency in Hz
    impedance : float
        Impedance in Ohm

    Returns
    -------
    float
        Total Electrical Energy Delivered in aJ (1e-18 J)
    '''

    teed = (amplitude ** 2 * pulse_width * stim_frequency * impedance)
    return teed


def compute_teed_from_amplitude_recording(amplitude: np.ndarray,
                                          tt: np.ndarray,
                                          pulse_width: int,
                                          f_stimulation: float,
                                          impedance: float) -> float:
    recording_length = tt.max() - tt.min()
    power = np.square(amplitude) * pulse_width * f_stimulation * impedance
    teed = np.trapz(power, tt) / recording_length
    return teed


def generate_amplitude_from_stim_periods(r: ingest.Recording,
                                         max_amplitude: int) -> np.ndarray:
    fs = 1 / r.dt
    n = r.electrode_data.shape[1]
    amplitude = np.zeros(n)
    for sp in r.stim_periods:
        stim_start = int(sp[0] * fs)
        stim_stop = int(sp[1] * fs)
        amplitude[stim_start:stim_stop] = max_amplitude
    return amplitude


def trim_recording(x: np.ndarray, fs: int,
                   slice_start: float,
                   slice_len: float) -> tuple[np.ndarray, np.ndarray]:
    slice_end = slice_start + slice_len
    slice_start_n = int(slice_start * fs)
    slice_end_n = int(slice_end * fs)
    trimmed = x[slice_start_n:slice_end_n]
    trimmed_tt = np.linspace(slice_start, slice_end, len(trimmed))
    return (trimmed, trimmed_tt)


def mean_stim_amplitude_from_gui_recording(rat_label, max_amplitude,
                                           datadir_gui):
    files = [f for f in datadir_gui.iterdir()
             if f.match('*' + rat_label + '*.txt')]
    amplitude_dict = dict()
    for f in files:
        recording_date = f.name.split('T')[0]
        a = max_amplitude[recording_date][rat_label]
        data = ingest.read_stim_amplitude_from_gui_recording(f, a)
        if recording_date in amplitude_dict:
            amplitude_dict[recording_date] = np.append(
                amplitude_dict[recording_date], data)
        else:
            amplitude_dict[recording_date] = data
        print_mean_and_percentile(f.name, data)
    all_amplitudes = []
    for d, a in amplitude_dict.items():
        s = '%s mean on %s' % (rat_label, d)
        data = amplitude_dict[d]
        all_amplitudes = np.append(all_amplitudes, data)
        print_mean_and_percentile(s, data)
    print_mean_and_percentile(rat_label + ' general mean', all_amplitudes)


def print_mean_and_percentile(intro_string, data):
    mean = np.mean(data)
    p10 = np.percentile(data, 10)
    p20 = np.percentile(data, 20)
    print('%s: %.2f uA, p10: %.2f, p20: %.2f' % (intro_string, mean, p10, p20))
