""" Test Generic Parser Functionality """
import os
import sys
import unittest
import importlib
import ast 

from src.data_handling.parsers.forty_two_parser import FortyTwo
from src.data_handling.parsers.csv_parser import CSV


class TestGenericParser(unittest.TestCase):

    def setUp(self):

        self.run_path = os.environ['RUN_PATH']

        self.rawDataFilepath = self.run_path + '/data/raw_telemetry_data/'
        self.tlmConfigFilepath = self.run_path + '/data/telemetry_configs/'

        self.parser_names = ['FortyTwo', 'CSV']
        self.parser_file_names = ['forty_two_parser', 'csv_parser']

        self.data_files = [str(['generic_test_42.txt']), str(['generic_test_csv.csv'])]
        self.config_files = [str(['generic_test_42_CONFIG.txt']), str(['generic_test_csv_CONFIG.txt'])]

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

            fileName = ast.literal_eval(self.data_files[i])[0]
            self.assertEquals(P.all_headers, { fileName : ['TIME', 'A', 'B', 'C']})
            self.assertEquals(P.sim_data, {'1000': { fileName : ['1000', '0', '0.000000000000e+00', '0.0']}, 
                                           '1001': { fileName : ['1001', '1', '1.000000000000e+00', '1.0']}, 
                                           '1002': { fileName : ['1002', '2', '2.000000000000e+00', '2']}, 
                                           '1003': { fileName : ['1003', '3', '3.000000000000e+00', '3']}})
            self.assertEquals(P.binning_configs, {'subsystem_assignments': {fileName: [['MISSION'], ['MISSION'], ['MISSION'], ['MISSION']]}, 
                                                  'test_assignments': {fileName: [[['SYNC', 'TIME']], [['FEASIBILITY', -1.0, 0.0, 10.0, 15.0]], [['NOOP']], [['FEASIBILITY', -1.0, 0.0, 10.0, 15.0]]]}, 
                                                  'description_assignments': {fileName: ['No description', 'No description', 'No description', 'No description']}})        


    def test_multisource_parse_sim_data(self):
        multi_source_data_files = [str(['generic_test_42.txt', 'generic_test_42_file2.txt'])]
        multi_source_config_files = [str(['generic_test_42_CONFIG.txt', 'generic_test_42_file2_CONFIG.txt'])]
        for i in range(len(self.parser_names)):
            parser = importlib.import_module('src.data_handling.parsers.' + self.parser_file_names[i])
            parser_class = getattr(parser, self.parser_names[i])
            P = parser_class(self.rawDataFilepath, 
                            self.tlmConfigFilepath,
                            multi_source_data_files[i],
                            multi_source_config_files[i])
            headers, sim_data, configs = P.get_sim_data()
            fileName1 = ast.literal_eval(multi_source_data_files[i])[0]
            fileName2 = ast.literal_eval(multi_source_data_files[i])[1]
            self.assertEquals(P.all_headers, { fileName1 : ['TIME', 'A', 'B', 'C'], fileName2 : ['TIME', 'D', 'E']})
            self.assertEquals(P.sim_data, {'1000': { fileName1 : ['1000', '0', '0.000000000000e+00', '0.0'], fileName2 : ['1000','0', '0.000000000000e+00']}, 
                                           '1001': { fileName1 : ['1001', '1', '1.000000000000e+00', '1.0']}, 
                                           '1002': { fileName1 : ['1002', '2', '2.000000000000e+00', '2']}, 
                                           '1003': { fileName1 : ['1003', '3', '3.000000000000e+00', '3'], fileName2 : ['1003','1', '1.000000000000e+00']},
                                           '1004': { fileName2 : ['1004','2', '2.000000000000e+00']}})
            self.assertEquals(P.binning_configs, {'subsystem_assignments': {fileName1: [['MISSION'], ['MISSION'], ['MISSION'], ['MISSION']], 
                                                                            fileName2 : [['MISSION'], ['MISSION'], ['MISSION']]}, 
                                                  'test_assignments': {fileName1: [[['SYNC', 'TIME']], [['FEASIBILITY', -1.0, 0.0, 10.0, 15.0]], [['NOOP']], [['FEASIBILITY', -1.0, 0.0, 10.0, 15.0]]], 
                                                                        fileName2 : [[['SYNC', 'TIME']], [['FEASIBILITY',-1.0, 0.0, 10.0, 15.0]], [['NOOP']]]}, 
                                                  'description_assignments': {fileName1: ['No description', 'No description', 'No description', 'No description'], 
                                                                                fileName2 : ['No description', 'No description', 'No description']}})
        
        
        


if __name__ == '__main__':
    unittest.main()
