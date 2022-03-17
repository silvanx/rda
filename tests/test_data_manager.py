import unittest
import shutil
import tempfile
from ratdata import data_manager as dm


class TestFindingRecordingFiles(unittest.TestCase):

    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)

    def test_empty_directory_contains_no_files(self):
        filelist = dm.find_all_recording_files_dir(self.test_dir)
        self.assertEqual(len(filelist), 0)

    def test_count_mce_files(self):
        test_files = ['2021-06-09T10-42-37 rat2 - OFT2.mat',
            '2021-06-24T10-17-47 rat5 CT.mat', 
            '2021-06-24T10-16-29 rat5 baseline.mat', 
            '2021-07-16T10-18-39 rat2 OFT2 random.mat', 
            '2021-07-16T09-58-23 rat2 baseline.mat'
            ]
        for filename in test_files:
            with open(self.test_dir + '/' + filename, 'w') as f:
                f.write('')
        filelist = dm.find_all_recording_files_dir(self.test_dir)
        self.assertEqual(len(filelist), 5)

    def test_count_gui_bin_files(self):
        test_files = ['2021-09-23T13-01-06-data.bin']
        for filename in test_files:
            with open(self.test_dir + '/' + filename, 'w') as f:
                f.write('')
        filelist = dm.find_all_recording_files_dir(self.test_dir)
        self.assertEqual(len(filelist), 1)

    def test_count_amplitude_files(self):
        test_files = ['2021-10-18T17-07-59-amplitude.txt',
                      '2021-10-18T17-22-10-amplitude.txt',
                      '2021-10-21T13-13-14-amplitude.txt',
                      '2021-10-29T11-06-51-amplitude.txt',
                      '2021-10-29T11-08-23-amplitude.txt',
                      '2021-11-02T11-18-33-amplitude.txt',
                      '2021-11-10T07-50-40-amplitude.txt',
                      '2021-11-10T07-56-09-amplitude.txt',
                      '2021-11-10T08-11-34-amplitude.txt',
                      '2021-11-10T08-17-27-amplitude.txt'
                      ]
        for filename in test_files:
            with open(self.test_dir + '/' + filename, 'w') as f:
                f.write('')
        filelist = dm.find_all_recording_files_dir(self.test_dir)
        self.assertEqual(len(filelist), 10)


if __name__ == '__main__':
    unittest.main()

