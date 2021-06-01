""" Test Generic Parser Functionality """
import os
import sys
import unittest
import importlib

from src.data_handling.parsers.forty_two_parser import FortyTwo

class TestGenericParser(unittest.TestCase):

    def setUp(self):

        self.run_path = os.environ['RUN_PATH']

        self.rawDataFilepath = self.run_path + '/data/raw_telemetry_data/'
        self.tlmConfigFilepath = self.run_path + '/data/telemetry_configs/'

        self.parser_names = ['FortyTwo']
        self.parser_file_names = ['forty_two_parser']

        self.data_files = [str(['generic_test_42.txt'])]
        self.config_files = [str(['generic_test_42_CONFIG.txt'])]

    # THESE THREE THINGS are needed by the time sync engine
    def test_singlesource_parse_sim_data(self):

        for i in range(len(self.parser_names)):
            parser = importlib.import_module('src.data_handling.parsers.' + self.parser_file_names[i])
            parser_class = getattr(parser, self.parser_names[i])
            P = parser_class(self.rawDataFilepath, 
                             self.tlmConfigFilepath,
                             self.data_files[i],
                             self.config_files[i])


            headers, sim_data, configs = P.get_sim_data()

            self.assertEquals(P.all_headers, {'generic_test_42.txt': ['TIME', 'A', 'B', 'C']})
            self.assertEquals(P.sim_data, {'1000': {'generic_test_42.txt': ['1000', '0', '0.000000000000e+00', '0.0']}, 
                                           '1001': {'generic_test_42.txt': ['1001', '1', '1.000000000000e+00', '1.0']}, 
                                           '1002': {'generic_test_42.txt': ['1002', '2', '2.000000000000e+00', '2']}, 
                                           '1003': {'generic_test_42.txt': ['1003', '3', '3.000000000000e+00', '3']}})
            self.assertEquals(P.binning_configs, {'subsystem_assignments': {'generic_test_42.txt': [['MISSION'], ['MISSION'], ['MISSION'], ['MISSION']]}, 
                                                  'test_assignments': {'generic_test_42.txt': [[['SYNC', 'TIME']], [['FEASIBILITY', -1.0, 0.0, 10.0, 15.0]], [['NOOP']], [['FEASIBILITY', -1.0, 0.0, 10.0, 15.0]]]}, 
                                                  'description_assignments': {'generic_test_42.txt': ['No description', 'No description', 'No description', 'No description']}})        


    # def test_multisource_parse_sim_data(self):
    #     multi_source_data_files = [str(['generic_test_42.txt', 'generic_test_42_file2.txt'])]
    #     multi_source_config_files = [str(['generic_test_42_CONFIG.txt', 'generic_test_42_file2_CONFIG.txt'])]
    #     for i in range(len(self.parser_names)):
    #         parser = importlib.import_module('src.data_handling.parsers.' + self.parser_file_names[i])
    #         parser_class = getattr(parser, self.parser_names[i])
    #         P = parser_class(self.rawDataFilepath, 
    #                          self.tlmConfigFilepath,
    #                          multi_source_data_files[i],
    #                          multi_source_config_files[i])
    #         headers, sim_data, configs = P.get_sim_data()


if __name__ == '__main__':
    unittest.main()
