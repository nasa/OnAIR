""" Test Execution Engine Functionality """
import os
import sys
import unittest
import shutil

from src.run_scripts.execution_engine import ExecutionEngine

## TODO: I think I need to set a global run path from the GET 

class TestExecutionEngine(unittest.TestCase):

    def setUp(self):
        self.test_path = os.path.dirname(os.path.abspath(__file__))
        self.config_fp = self.test_path + '/../data/config/default_config.ini'
        
        self.E = ExecutionEngine('', 'test', False)
        self.save_path = os.environ['RESULTS_PATH']


        # self.tmp_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)) + '/../../../', 'results/tmp')
        self.tmp_save_path = os.environ['RESULTS_PATH'] + '/tmp'

    def test_init_execution_engine(self):

        # Init Housekeeping 
        self.assertEquals(self.E.run_name,'test')

        # Init Flags 
        self.assertEquals(self.E.IO_Flag, False)
        self.assertEquals(self.E.Dev_Flag, False)
        self.assertEquals(self.E.SBN_Flag, False)
        self.assertEquals(self.E.Viz_Flag, False)
        
        # Init Paths 
        self.assertEquals(self.E.dataFilePath, '')
        self.assertEquals(self.E.metadataFilePath, '')
        self.assertEquals(self.E.benchmarkFilePath, '')
        self.assertEquals(self.E.metaFiles, '')
        self.assertEquals(self.E.telemetryFiles, '')
        self.assertEquals(self.E.benchmarkFiles, '')

        # Init parsing/sim info
        self.assertEquals(self.E.parser_file_name, '')
        self.assertEquals(self.E.parser_name, '')
        self.assertEquals(self.E.sim_name, '')
        self.assertEquals(self.E.data, None)
        self.assertEquals(self.E.binnedData, None)
        self.assertEquals(self.E.sim, None)

        self.assertEquals(self.E.save_flag, False)
        self.assertEquals(self.E.save_name, 'test')

    # Note: the results directory AND tmp subdir needs to already exist
    # (maybe do a check and if the dir isnt there, make it)
    def test_init_save_paths(self):
        self.E.init_save_paths()
        sub_dirs = os.listdir(self.tmp_save_path)
        self.assertEquals(sub_dirs, ['associativity', 'diagnosis', 'graphs', 'models', 'tensorboard', 'viz'])
        
        # Try to init again now that tmp exits 
        self.E.init_save_paths()

        sub_dirs = os.listdir(self.tmp_save_path)
        self.assertEquals(sub_dirs, ['associativity', 'diagnosis', 'graphs', 'models', 'tensorboard', 'viz'])
        
    def test_delete_save_paths(self):
        self.E.delete_save_paths()
        sub_dirs = os.listdir(self.save_path)
        self.assertEquals(sub_dirs, [])

    def test_parse_configs(self):
        self.E.parse_configs(self.config_fp)
        self.assertEquals(self.E.dataFilePath, '/src/data/raw_telemetry_data/')
        self.assertEquals(self.E.metadataFilePath, '/src/data/telemetry_configs/')
        self.assertEquals(self.E.metaFiles, "['42_TLM_CONFIG.txt']")
        self.assertEquals(self.E.telemetryFiles, "['42_TLM.txt']")

        # No benchmarks passed in this case 
        self.assertEquals(self.E.benchmarkFilePath, '')
        self.assertEquals(self.E.benchmarkFiles, '')
        self.assertEquals(self.E.benchmarkIndices, '')

        self.assertEquals(self.E.parser_file_name, 'forty_two_parser')
        self.assertEquals(self.E.parser_name, 'FortyTwo')
        self.assertEquals(self.E.sim_name, 'FortyTwo')
        
        self.assertEquals(self.E.IO_Flag, False)
        self.assertEquals(self.E.Dev_Flag, False)
        self.assertEquals(self.E.SBN_Flag, False)
        self.assertEquals(self.E.Viz_Flag, False)

    def test_parse_data(self):
        parser_name = 'FortyTwo'
        parser_file_name = 'forty_two_parser'
        dataFilePath = '/data/raw_telemetry_data/'
        metadataFilePath = '/data/telemetry_configs/'

        self.E.telemetryFiles = "['42_TLM.txt']"
        self.E.metaFiles = "['42_TLM_CONFIG.txt']"

        self.E.parse_data(parser_name, parser_file_name, dataFilePath, metadataFilePath)
        
        self.assertTrue(self.E.data)

        # parser = importlib.import_module('src.data_handling.parsers.' + parser_file_name)
        # parser_class = getattr(parser, parser_name) # This could be simplified if the parsers all extend a parser class... but this works for now
        # tm_data_path = os.path.dirname(os.path.realpath(__file__)) + '/../..' + dataFilePath
        # tm_metadata_path = os.path.dirname(os.path.realpath(__file__)) + '/../..' + metadataFilePath
        # self.data = parser_class(tm_data_path, tm_metadata_path, self.telemetryFiles, self.metaFiles)

    def test_run_sim(self):
        return 

    def test_save_results(self):
        return 

    def test_set_run_param(self):
        return 


if __name__ == '__main__':
    unittest.main()
