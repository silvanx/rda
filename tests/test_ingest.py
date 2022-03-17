from unittest.mock import patch, mock_open
import unittest
import numpy as np
from ratdata import ingest


class TestDataCleaning(unittest.TestCase):

    def test_removing_missing_full_hs_samples_not_on_boundary(self):
        bad_data = np.array([
            [1, 2, 3, 4],
            [1e7, 1e7, 1e7, 1e7],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4]
        ])
        good_data = np.array([
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4]
        ])

        cleaned_data = ingest.replace_oob_samples(bad_data)
        np.testing.assert_array_equal(good_data, cleaned_data)

    def test_removing_missing_full_hs_samples_first_sample(self):
        bad_data = np.array([
            [1e7, 1e7, 1e7, 1e7],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4]
        ])
        good_data = np.array([
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4]
        ])
        cleaned_data = ingest.replace_oob_samples(bad_data)
        np.testing.assert_array_equal(good_data, cleaned_data)

    def test_removing_missing_full_hs_samples_last_sample(self):
        bad_data = np.array([
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1e7, 1e7, 1e7, 1e7]
        ])
        good_data = np.array([
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4]
        ])
        cleaned_data = ingest.replace_oob_samples(bad_data)
        np.testing.assert_array_equal(good_data, cleaned_data)

    def test_removing_missing_random_hs_samples(self):
        bad_data = np.array([
            [1, 2, -1e7, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 1e7],
            [1, -1e7, 3, 4],
            [1, 2, 3, 4],
            [1e7, 2, 3, 4]
        ])
        good_data = np.array([
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4]
        ])
        cleaned_data = ingest.replace_oob_samples(bad_data)
        np.testing.assert_array_equal(good_data, cleaned_data)


class TestIngestMceMatlabFile(unittest.TestCase):

    def test_extract_info_from_filename(self):
        filename = '2021-07-26T10-19-22 rat1 baseline.mat'
        datetime = '2021-07-26T10-19-22'
        rat = 'rat1'
        condition = 'baseline'

        res = ingest.extract_info_from_filename(filename)
        self.assertEqual(datetime, res[0])
        self.assertEqual(rat, res[1])
        self.assertEqual(condition, res[2])


class TestIngestGuiAmplitude(unittest.TestCase):

    def setUp(self) -> None:
        self.amplitude_mock_file = 'file/mock/amplitude.txt'
        self.amplitude_data_txt = """--- START RECORDING: 2021-11-02T10-49-08.581 ---
9
9
9
9
9
8
8
8
8
8
8
15
15
15
15
15
10
10
10
10
10
10
10
10
10
10
10
14
14
14
14
14
14
6
6
6
6
6
8
8
8
8
8
8
8
8
8
8
8
7
7
7
7
7
--- END RECORDING: 2021-11-02T10-49-09.111 ---"""

    def test_load_amplitude_return_type(self):
        with patch('ratdata.ingest.open', new=mock_open(read_data=self.amplitude_data_txt)) as f:
            amplitude_data = ingest.read_gui_amplitude_file_data(self.amplitude_mock_file)
            f.assert_called_once_with(self.amplitude_mock_file, 'r')

        self.assertIsInstance(amplitude_data, np.ndarray)

    def test_load_amplitude_array_dimension(self):
        with patch('ratdata.ingest.open', new=mock_open(read_data=self.amplitude_data_txt)) as f:
            amplitude_data = ingest.read_gui_amplitude_file_data(self.amplitude_mock_file)
            f.assert_called_once_with(self.amplitude_mock_file, 'r')

        self.assertEqual((54,), amplitude_data.shape)


if __name__ == '__main__':
    unittest.main()
