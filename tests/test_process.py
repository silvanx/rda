from unittest.mock import patch, mock_open
import unittest

from numpy.testing import assert_almost_equal
from ratdata import process, ingest
import numpy as np

class TestDataProcessing(unittest.TestCase):

    def setUp(self) -> None:
        self.fs = 20000
        self.target_fs = 500
        self.tt = np.arange(0, 1, 1 / self.fs)
        self.sine_20 = np.sin(2 * 20 * np.pi * self.tt)
        self.ttd = np.arange(0, 1, 1 / self.target_fs)
        self.sine_20d = np.sin(2 * 20  * np.pi * self.ttd)

    def test_signal_downsampled_length(self):
        decimated = process.downsample_signal(self.sine_20, self.fs, self.target_fs)
        self.assertEqual(decimated.shape, (500,))

    def test_signal_downsampled_output(self):
        decimated = process.downsample_signal(self.sine_20, self.fs, self.target_fs)
        np.testing.assert_almost_equal(decimated, self.sine_20d, decimal=1)

    def test_bandpass_filter_pass(self):
        numtaps = 201
        filtered = process.bandpass_filter(self.sine_20d, self.target_fs, 1, 50, numtaps)
        np.testing.assert_almost_equal(filtered[numtaps:], self.sine_20d[numtaps:], decimal=1)

    def test_bandpass_filter_reject(self):
        filtered = process.bandpass_filter(self.sine_20d, self.target_fs, 50, 100)
        target = np.zeros(filtered.shape)
        np.testing.assert_almost_equal(filtered, target, decimal=1)
        
    def test_rolling_power_constant_signal(self):
        constant_signal = np.ones(20000)
        segment_length = 100
        power = process.rolling_power_signal(constant_signal, segment_length)
        cut = int(segment_length / 2)
        np.testing.assert_almost_equal(constant_signal[cut:-cut], power[cut:-cut], decimal=2)

    def test_band_power_empty_band(self):
        power = process.power_in_frequency_band(self.sine_20d, 80, 100, self.target_fs)
        self.assertAlmostEqual(power, 0.0)

    def test_band_power_full_spectrum(self):
        power = process.power_in_frequency_band(self.sine_20d, 0, 250, self.target_fs)
        self.assertAlmostEqual(power, np.sum(self.sine_20d ** 2 / self.target_fs))

    def test_relative_beta_pure_sine(self):
        relative_power = process.compute_relative_power(self.sine_20d, 17, 23, self.target_fs)
        self.assertAlmostEqual(relative_power, 1.0)

    def test_smaller_range_smaller_power(self):
        power_small = process.compute_relative_power(self.sine_20d, 0, 10, self.target_fs)
        power_medium = process.compute_relative_power(self.sine_20d, 0, 50, self.target_fs)
        power_big = process.compute_relative_power(self.sine_20d, 0, 250, self.target_fs)
        power_big_without_dc = process.compute_relative_power(self.sine_20d, 1, 250, self.target_fs)
        self.assertLessEqual(power_small, power_medium)
        self.assertLessEqual(power_medium, power_big)
        self.assertLessEqual(power_big_without_dc, power_big)

    def test_power_change(self):
        s20d_high_amplitude = 2 * self.sine_20d
        s20d_low_amplitude = 0.5 * self.sine_20d
        change_high = process.compute_change_in_power(self.sine_20d, s20d_high_amplitude, 17, 23, self.target_fs)
        change_low = process.compute_change_in_power(self.sine_20d, s20d_low_amplitude, 17, 23, self.target_fs)
        self.assertAlmostEqual(change_high, 300)
        self.assertAlmostEqual(change_low, -75)

    def test_teed_from_amplitude_recording_constant_amplitude(self):
        n = 400
        f = 200
        a = 100
        pw = 80
        impedance = 1
        f_stimulation = 130
        amplitude = a * np.ones(n)
        tt = np.linspace(0, n/f, n)
        teed_analytical = process.compute_teed_continuous_stim(a, pw, f_stimulation, impedance)
        teed_from_data = process.compute_teed_from_amplitude_recording(amplitude, tt, pw, f_stimulation, impedance)
        self.assertEqual(teed_analytical, teed_from_data)
    
    def test_generate_stim_amplitude_from_stim_periods(self):
        f = 20000
        max_amplitude = 100
        stim_periods = ((1, 2),)
        r = ingest.Recording(np.zeros((4, 3 * f)), 'dummy filename', 1 / f, None, 'ratX', 'dummy', stim_periods)
        stim_amplitude_data = process.generate_amplitude_from_stim_periods(r, max_amplitude)
        self.assertEqual(len(stim_amplitude_data), 3 * f)
        self.assertEqual(np.sum(stim_amplitude_data), max_amplitude * f)
        self.assertEqual(np.sum(stim_amplitude_data[:f]), 0)
        self.assertEqual(np.sum(stim_amplitude_data[-f:]), 0)
        
    def test_trim_recording(self):
        fs = 20000
        max_t = 5
        slice_start = 1
        slice_length = 2
        slice_end = slice_start + slice_length
        data = np.zeros(max_t * fs)
        data[:int(fs * (slice_start + slice_length / 2))] = 1 / (slice_length * fs)
        data[int(slice_end * fs):] = 1
        trimmed, tt = process.trim_recording(data, fs, slice_start, slice_length)
        self.assertEqual(min(tt), slice_start)
        self.assertEqual(max(tt), slice_end)
        self.assertEqual(len(tt), len(trimmed))
        self.assertEqual(len(tt), slice_length * fs)
        self.assertAlmostEqual(sum(trimmed), 0.5)
        
    

if __name__ == '__main__':
    unittest.main()
