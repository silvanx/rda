
import scipy.signal as signal
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import numpy as np
from ratdata import data_manager as dm, ingest, process
import pywt


def resample_interpolate(data: np.ndarray, fs: float,
                         target_fs: float) -> np.ndarray:
    n = len(data)
    t_max = n / fs
    ratio = fs / target_fs
    tt1 = np.linspace(0, t_max, n)
    tt2 = np.linspace(0, t_max, int(n/ratio))
    b, a = signal.butter(6, 2 * np.pi / ratio, btype='low', analog=False)
    filtered = signal.filtfilt(b, a, data)
    interpolated = interpolate.interp1d(tt1, filtered, kind='cubic')
    return interpolated(tt2)


def beta_envelope(data: np.ndarray, fs: float, target_fs: float = 300
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    downsampled_data = resample_interpolate(data, fs, target_fs)
    filtered = highpass_filter(downsampled_data, target_fs, 1)
    if target_fs != 300:
        raise NotImplementedError
    # ~14-~18 Hz
    # scales = np.linspace(13.55, 17.4, 8)
    # ~13-~30 Hz
    scales = np.linspace(8.15, 18.75, 34)
    wavelets, _ = pywt.cwt(filtered, scales, 'morl',
                           sampling_period=(1/target_fs))
    averaged = np.mean(wavelets, axis=0)
    rectified = np.abs(averaged)
    smoothed_rectified = np.convolve(rectified, np.ones(20)) / 20
    analytic = signal.hilbert(averaged)
    envelope = np.abs(analytic)
    smoothed_envelope = np.convolve(envelope, np.ones(20)) / 20
    return (averaged, smoothed_envelope, smoothed_rectified)


def beta_bursts(envelope: np.ndarray,
                threshold: float) -> list[tuple[int, int, float]]:
    beta_r = envelope - threshold
    crossings = np.where(np.diff(np.sign(beta_r)))[0]
    bursts = []
    last_start = None
    last_end = None
    for c in crossings:
        if beta_r[c] < 0 and beta_r[c + 1] > 0:
            last_start = c
        if beta_r[c] > 0 and beta_r[c + 1] < 0 and last_start is not None:
            last_end = c
            amplitude = np.max(beta_r[last_start: last_end])
            bursts.append((last_start, last_end, amplitude))
    return bursts


def power_in_frequency_band(data: np.ndarray, low: float, high: float,
                            fs: int) -> float:
    f, pxx = signal.welch(data, fs, nperseg=2 * fs)
    idx = np.logical_and(f >= low, f <= high)
    f_res = f[1] - f[0]
    power = integrate.trapz(pxx[idx], dx=f_res)
    return power


def oof_power_in_frequency_band(m: float, b: float,
                                low: float, high: float) -> float:
    return np.e ** b / (m + 1) * (high ** (m + 1) - low ** (m + 1))


def find_peaks(f: np.ndarray, psd: np.ndarray,
               f_low: float, f_high: float) -> list[tuple[float, float]]:
    idx = np.where(np.logical_and(f >= f_low, f <= f_high))[0]
    peaks, _ = signal.find_peaks(psd[idx])
    prominences, _, _ = signal.peak_prominences(psd[idx], peaks)
    peak_locations = idx[peaks]
    peak_list = [(f[l], psd[l], prominences[i])
                 for i, l in enumerate(peak_locations)]
    return sorted(peak_list, key=lambda e: e[2], reverse=True)


def power_in_band_no_oof(data: np.ndarray, low: float, high: float, fs: int,
                         oof_low: float, oof_high: float, scale: float):
    f, pxx = signal.welch(data, fs, nperseg=fs)
    idx = np.logical_and(f >= low, f <= high)
    f_res = f[1] - f[0]
    m, b = process.fit_oof(f, pxx, oof_low, oof_high)
    oof_power = oof_power_in_frequency_band(m, b, low, high)
    power = integrate.trapz(pxx[idx], dx=f_res)
    return power - oof_power


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


def highpass_filter(data: np.ndarray, fs: int, cutoff: float,
                    numtaps: int = 201) -> np.ndarray:
    nyq = fs / 2
    cutoff = cutoff / nyq
    taps = signal.firwin(numtaps, cutoff, pass_zero='highpass')
    filtered = signal.filtfilt(taps, 1.0, data)
    return filtered


def lowpass_filter(data: np.ndarray, fs: int, cutoff: float,
                   numtaps: int = 201) -> np.ndarray:
    nyq = fs / 2
    cutoff = cutoff / nyq
    taps = signal.firwin(numtaps, cutoff, pass_zero='lowpass')
    filtered = signal.filtfilt(taps, 1.0, data)
    return filtered


def interpolated_highpass_filter(data: np.ndarray, fs: int, q: int,
                                 cutoff: float,
                                 numtaps: int = 201) -> np.ndarray:
    low_fs = int(fs / q)
    decimated = signal.decimate(data, q)
    low = lowpass_filter(decimated, low_fs, cutoff, numtaps)
    ttd = np.linspace(0, 1, low.shape[-1])
    tt = np.linspace(0, 1, data.shape[-1])
    if len(data.shape) == 1:
        filtered = interpolate.interp1d(ttd, low)(tt)
    else:
        filtered = np.zeros(data.shape)
        for i in range(data.shape[0]):
            filtered[i, :] = interpolate.interp1d(ttd, low[i, :])(tt)
    return data - filtered


def compute_relative_power(data: np.ndarray, low: int, high: int,
                           fs: int, total_low: int = 0,
                           total_high: int = None) -> float:
    if not total_high:
        total_high = fs / 2
    band_power = power_in_frequency_band(data, low, high, fs)
    total_power = power_in_frequency_band(data, total_low, total_high, fs)
    relative_power = band_power / total_power
    return relative_power


def rolling_power_signal(data: np.ndarray, segment_len: int) -> np.ndarray:
    d2 = data ** 2
    power = np.convolve(d2, np.ones(segment_len) / segment_len, mode='same')
    return power


def compute_change_in_power(data1: np.ndarray, data2: np.ndarray, low: int,
                            high: int, fs: int) -> float:
    power1 = power_in_frequency_band(data1, low, high, fs)
    power2 = power_in_frequency_band(data2, low, high, fs)
    return (power2 / power1 - 1) * 100


def get_change_in_beta_power_from_rec(rec: dm.RecordingFile,
                                      remove_oof: bool = False) -> float:
    if rec.baseline.count() == 0:
        return None

    baseline = rec.baseline.get().baseline
    recording_power = dm.RecordingPower.get(recording=rec)
    baseline_power = dm.RecordingPower.get(recording=baseline)
    if remove_oof:
        rpower = recording_power.beta_power_without_oof
        bpower = baseline_power.beta_power_without_oof
    else:
        rpower = recording_power.beta_power
        bpower = baseline_power.beta_power
    power_change = (rpower / bpower - 1) * 100
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
        try:
            a = max_amplitude[recording_date][rat_label]
        except KeyError:
            print('No amplitude data for %s' % f.name)
            continue
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


def print_mean_and_percentile(intro_string: str, data: np.ndarray) -> None:
    mean = np.mean(data)
    p10 = np.percentile(data, 10)
    p20 = np.percentile(data, 20)
    print('%s: %.2f uA, p10: %.2f, p20: %.2f' % (intro_string, mean, p10, p20))


def pulses_in_slice(p: tuple[float, float], fs: int,
                    slice_start: int, slice_end: int) -> bool:
    return p[0] * fs > slice_start and p[1] * fs < slice_end


def fit_oof(f: np.ndarray, pxx: np.ndarray, f_min: float,
            f_max: float) -> tuple[float, float]:
    idx = np.where((f >= f_min) & (f <= f_max))
    m, b = np.polyfit(np.log(f[idx]), np.log(pxx[idx]), 1)
    return m, b


def create_pulse_template(rec: ingest.Recording,
                          template_length: int = None,
                          start_offset: int = 0,
                          align: str = 'start',
                          slice: tuple[float, float] = None,
                          channels: str = 'mean',
                          highpass_cutoff: float = None) -> np.ndarray:
    fs = int(1 / rec.dt)
    if slice is None:
        slice_start = 0
        slice_end = rec.electrode_data.shape[1]
    else:
        slice_start = int(slice[0] * fs)
        slice_end = int(slice[1] * fs)

    if template_length is None:
        longest_pulse = max(rec.pulse_periods,
                            key=lambda item: item[1] - item[0])
        template_length = int((longest_pulse[1] - longest_pulse[0]) * fs)

    if channels == 'mean':
        prepared_data = np.mean(rec.electrode_data,
                                axis=0)[slice_start:slice_end]
    elif channels == 'all':
        prepared_data = rec.electrode_data[:, slice_start:slice_end]

    if highpass_cutoff is not None:
        prepared_data = interpolated_highpass_filter(prepared_data, fs, 40,
                                                     highpass_cutoff)

    pulses = list(filter(lambda p: pulses_in_slice(p, fs,
                                                   slice_start, slice_end),
                         rec.pulse_periods))

    n_channels = prepared_data.shape[0] if\
        len(prepared_data.shape) == 2 else 1
    n_samples = prepared_data.shape[1] if\
        len(prepared_data.shape) == 2 else len(prepared_data)
    template = np.zeros((n_channels, template_length))
    skipped_pulses = 0
    if len(pulses) > 0:
        for s, e in pulses:
            half_t_length = int(np.floor(template_length / 2))
            for i in range(n_channels):
                if n_channels > 1:
                    s_n = int(s * fs) - slice_start + start_offset
                    e_n = s_n + template_length
                    if align == 'max':
                        d = prepared_data[i, s_n:e_n]
                        if len(d) > 0:
                            max_location = np.argmax(d)
                            s_n = s_n + max_location - half_t_length
                            e_n = s_n + template_length
                    if s_n < 0 or e_n > n_samples or s_n > n_samples:
                        skipped_pulses += 1
                    else:
                        template[i, :] += prepared_data[i, s_n: e_n]
                else:
                    s_n = int(s * fs) - slice_start + start_offset
                    e_n = s_n + template_length
                    if align == 'max':
                        d = prepared_data[s_n:e_n]
                        if len(d) > 0:
                            max_location = np.argmax(d)
                            s_n = s_n + max_location - half_t_length
                            e_n = s_n + template_length
                    if s_n < 0 or e_n > n_samples or s_n > n_samples:
                        skipped_pulses += 1
                    else:
                        template[i, :] += prepared_data[s_n: e_n]
        template /= (len(pulses) - skipped_pulses)
    if template.shape[0] == 1:
        template = template.flatten()
    return template


def compute_percentage_time_on(amplitude_data: np.ndarray) -> float:
    samplitude = np.sort(amplitude_data)
    index = np.where(samplitude > 0)[0][0]
    return 1 - index / len(samplitude)


def linear_interpolate(start: float, end: float, length: int) -> np.ndarray:
    dx = (end - start) / (length - 1)
    return np.array([start + i * dx for i in range(length)])


def dsp_filter(b: np.ndarray, a: np.ndarray, filter_length: int,
               x_previous: np.ndarray, y_previous: np.ndarray,
               x_current: float):
    if filter_length <= 0:
        return
    b_acc = b[0] * x_current
    a_acc = 0
    for i in range(1, filter_length):
        b_acc += b[i] * x_previous[i - 1]
        a_acc += a[i] * y_previous[i - 1]
    y_current = b_acc - a_acc
    y_previous = np.roll(y_previous, 1)
    x_previous = np.roll(x_previous, 1)
    y_previous[0] = y_current
    x_previous[0] = x_current
    return x_previous, y_previous


def dsp_biomarker(data: np.ndarray) -> np.ndarray:
    # First downsampling (Butterworth, lowpass, cutoff 0.1 (normalized))
    a1 = [1, -2.37409474370935, 1.92935566909122, -0.532075368312092]
    b1 = [0.00289819463372137, 0.00869458390116412, 0.00869458390116412,
          0.00289819463372137]
    decimation_factor1 = 10
    # Second downsampling (Butterworth, lowpass, cutoff 0.25 (normalized))
    a2 = [1, -1.45902906222806, 0.910369000290069, -0.197825187264319]
    b2 = [0.0316893438497110, 0.0950680315491330, 0.0950680315491330,
          0.0316893438497110]
    decimation_factor2 = 4
    # Beta extraction (Butterworth, bandpass, 13-30 Hz, assume fs = 500 Hz)
    a3 = [1, -5.40212898354275, 12.3252130462259, -15.1987442456715,
          10.6838109529814, -4.05972313481503, 0.651760197122122]
    b3 = [0.000995173561555180, 0, -0.00298552068466554, 0,
          0.00298552068466554, 0, -0.000995173561555180]

    total_decimation = decimation_factor1 * decimation_factor2
    lowpass_len = len(a1)
    bandpass_len = len(a3)
    y_previous1 = np.zeros(lowpass_len)
    x_previous1 = np.zeros(lowpass_len)
    y_previous2 = np.zeros(lowpass_len)
    x_previous2 = np.zeros(lowpass_len)
    y_previous3 = np.zeros(bandpass_len)
    x_previous3 = np.zeros(bandpass_len)
    power_segment_len = 50

    history_len = int(np.ceil(len(data) / total_decimation))
    full_beta_history = np.zeros(history_len)
    full_total_history = np.zeros(history_len)
    beta_power = np.zeros(len(data))
    total_power = np.zeros(len(data))

    for i, x_current in enumerate(data):
        x_previous1, y_previous1 = dsp_filter(b1, a1, lowpass_len, x_previous1,
                                              y_previous1, x_current)
        y_current1 = y_previous1[0]
        if i % decimation_factor1 == 0:
            decimated1 = y_current1
            x_previous2, y_previous2 = dsp_filter(b2, a2, lowpass_len,
                                                  x_previous2, y_previous2,
                                                  decimated1)
            y_current2 = y_previous2[0]
            if i % total_decimation == 0:
                decimated2 = y_current2
                x_previous3, y_previous3 = dsp_filter(b3, a3, bandpass_len,
                                                      x_previous3, y_previous3,
                                                      decimated2)
                y_current3 = y_previous3[0]
                history_index = int(i / total_decimation)
                full_beta_history[history_index] = y_current3
                full_total_history[history_index] = decimated2
    power_window = np.ones(power_segment_len) / (power_segment_len)
    beta_power = np.convolve(full_beta_history ** 2, power_window, 'same')
    total_power = np.convolve(full_total_history ** 2, power_window, 'same')
    biomarker = 1000 * beta_power / total_power
    return biomarker, beta_power, total_power


def subtract_template(data, template, blank=False, interpolate=True):
    align = template.align
    blank_template = np.zeros(template.template.shape)
    half_template_length = int(np.floor(template.length / 2))
    if template.channels == 1:
        data = np.mean(data, axis=0)
        for s in template.start:
            e = s + template.length
            if e < len(data):
                if align == 'max':
                    d = data[s:e]
                    if len(d) > 0:
                        max_location = np.argmax(data[s:e])
                        s_n = s + max_location - half_template_length
                        e_n = s_n + template.length
                    if s_n > 0 and e_n < len(data) and len(d) > 0:
                        if blank:
                            data[s_n:e_n] = (
                                linear_interpolate(data[s_n], data[e_n],
                                                   template.length)
                                if interpolate else blank_template)
                        else:
                            data[s_n:e_n] -= template.template
                else:
                    if blank:
                        data[s:e] = (
                            linear_interpolate(data[s], data[e],
                                               template.length)
                            if interpolate else blank_template)
                    else:
                        data[s:e] -= template.template
    elif template.channels > 1:
        for s in template.start:
            e = s + template.length
            if e < data.shape[1]:
                if align == 'max':
                    for i in range(data.shape[0]):
                        d = data[i, s:e]
                        if len(d) > 0:
                            max_location = np.argmax(d)
                            s_n = s + max_location - half_template_length
                            e_n = s_n + template.length
                        if s_n > 0 and e_n < data.shape[1] and len(d) > 0:
                            if blank:
                                data[i, s_n:e_n] = (
                                    linear_interpolate(data[i, s_n],
                                                       data[i, e_n],
                                                       template.length)
                                    if interpolate else blank_template)
                            else:
                                data[i, s_n:e_n] -= template.template[i, :]
                else:
                    if blank:
                        data[:, s:e] = (
                            linear_interpolate(data[:, s],
                                               data[:, e],
                                               template.length).T
                            if interpolate else blank_template)
                    else:
                        data[:, s:e] -= template.template
    return data
