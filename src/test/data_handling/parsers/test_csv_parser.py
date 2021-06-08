""" Test CSV Parser Functionality """
import os
import sys
import unittest
import shutil

from src.data_handling.parsers.forty_two_parser import CSV

class TestCSVParser(unittest.TestCase):

    def setUp(self):
        self.P = CSV()
        self.run_path = os.environ['RUN_PATH']
        self.rawDataFilepath = self.run_path + '/data/raw_telemetry_data/'
        self.tlmConfigFilepath = self.run_path + '/data/telemetry_configs/'

    def test_init_empty_parser(self):
        self.assertEquals(self.P.raw_data_file_path, '')
        self.assertEquals(self.P.metadata_file_path, '')
        self.assertEquals(self.P.all_headers, '')
        self.assertEquals(self.P.sim_data, '')
        self.assertEquals(self.P.binning_configs, '')

    def test_init_nonempty_parser(self):
        P = CSV(self.run_path + '/data/raw_telemetry_data/', 
                            self.run_path + '/data/telemetry_configs/',
                            str(['nominal0.csv']),
                            str(['nominal0_config.txt']))

        self.assertEquals(P.raw_data_file_path, self.run_path + '/data/raw_telemetry_data/')
        self.assertEquals(P.metadata_file_path, self.run_path + '/data/telemetry_configs/')
                            

                            
if __name__ == '__main__':
    unittest.main()