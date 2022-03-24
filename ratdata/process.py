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
