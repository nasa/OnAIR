""" Test CSV Parser Functionality """
import os
import sys
import unittest
import shutil

from src.data_handling.parsers.csv_parser import CSV

class TestCSVParser(unittest.TestCase):

    def setUp(self):
        self.P = CSV()
        self.run_path = os.environ['RUN_PATH']
        self.rawDataFilepath = self.run_path + '/data/raw_telemetry_data/'
        self.tlmConfigFilepath = self.run_path + '/data/telemetry_configs/'

    def test_init_empty_parser(self):
        self.assertEqual(self.P.raw_data_file_path, '')
        self.assertEqual(self.P.metadata_file_path, '')
        self.assertEqual(self.P.all_headers, '')
        self.assertEqual(self.P.sim_data, '')
        self.assertEqual(self.P.binning_configs, '')

    def test_init_nonempty_parser(self):
        P = CSV(self.run_path + '/data/raw_telemetry_data/', 
                            self.run_path + '/data/telemetry_configs/',
                            str(['data1.csv']),
                            str(['data1_CONFIG.txt']))

        self.assertEqual(P.raw_data_file_path, self.run_path + '/data/raw_telemetry_data/')
        self.assertEqual(P.metadata_file_path, self.run_path + '/data/telemetry_configs/')
                      
if __name__ == '__main__':
    unittest.main()