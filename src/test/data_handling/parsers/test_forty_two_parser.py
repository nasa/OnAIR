""" Test 42 Parser Functionality """
import os
import sys
import unittest
import shutil

from src.data_handling.parsers.forty_two_parser import FortyTwo

class TestFortyTwoParser(unittest.TestCase):

    def setUp(self):
        self.P = FortyTwo()
        self.run_path = os.environ['RUN_PATH']
        self.rawDataFilepath = self.run_path + '/data/raw_telemetry_data/'
        self.tlmConfigFilepath = self.run_path + '/data/telemetry_configs/'

    def test_init_parser(self):
        self.assertEquals(self.P.raw_data_file_path, '')
        self.assertEquals(self.P.metadata_file_path, '')
        self.assertEquals(self.P.all_headers, '')
        self.assertEquals(self.P.sim_data, '')
        self.assertEquals(self.P.binning_configs, '')

    def test_parse_sim_data(self):
        dataFiles = '42_TLM.txt'
        self.P.raw_data_file_path = self.rawDataFilepath
        hdrs, data = self.P.parse_sim_data(dataFiles) # Can only parse one datafile!  

        # Test 'hdrs'
        parsed_filenames = list(hdrs.keys())                       # Just one file parsed
        self.assertEquals(len(parsed_filenames), 1)                # .
        self.assertEquals(parsed_filenames[0], '42_TLM.txt')       # ..
        self.assertEquals(len(hdrs['42_TLM.txt']), 13)             # 13 TLM points (data fields) 

        # Test 'data'
        self.assertEquals(len(data.values()), 5)                   # 5 time steps 
        for frame in data.values():                                # Parsing for timesteps formatted correctly
            self.assertEquals(list(frame.keys())[0], '42_TLM.txt') # 13 tlm points in each frame 
            self.assertEquals(len(frame['42_TLM.txt']), 13)        # .

        # Test time ordering
        self.assertEquals(list(data.keys()), ['2019-127-12:00:17.300006746', 
                                              '2019-127-12:00:17.300006747', 
                                              '2019-127-12:00:17.300006748', 
                                              '2019-127-12:00:17.300006790', 
                                              '2019-127-12:00:17.300006801'])
        # TODO: Need more time sync testing 

    def test_parse_headers(self):
        txt_file = open(self.rawDataFilepath + 'single_frame.txt',"r+")

        data_str = txt_file.read().split('\n[EOF]')[0]
        txt_file.close()

        hdrs = self.P.parse_headers(data_str)

        self.assertEquals(len(hdrs), 13)
        self.assertEquals(hdrs, ['TIME', 'SAMPLE.sample_data_tlm_t.sample_data_counter', 'SAMPLE.sample_data_tlm_t.sample_data_value', 'SAMPLE.sample_data_power_t.sample_data_counter', 'SAMPLE.sample_data_power_t.sample_data_voltage', 'SAMPLE.sample_data_power_t.sample_data_current', 'SAMPLE.sample_data_thermal_t.sample_data_counter', 'SAMPLE.sample_data_thermal_t.sample_data_internal_temp', 'SAMPLE.sample_data_thermal_t.sample_data_external_temp', 'SAMPLE.sample_data_gps_t.sample_data_counter', 'SAMPLE.sample_data_gps_t.sample_data_lat', 'SAMPLE.sample_data_gps_t.sample_data_lng', 'SAMPLE.sample_data_gps_t.sample_data_alt'])


    def test_parse_frame(self):
        txt_file = open(self.rawDataFilepath + 'single_frame.txt',"r+")
        data_str = txt_file.read().split('\n[EOF]')[0]
        txt_file.close()
        clean_frame = self.P.parse_frame(data_str) 
        self.assertEquals(clean_frame, ['2019-127-12:00:17.300006801', '4', '0.000000000000e+00', '0', '0.000000000000e+00', '0.000000000000e+00', '0', '0.000000000000e+00', '0.000000000000e+00', '0', '0.000000000000e+00', '0.000000000000e+00', '0.000000000000e+00'])

    def test_parse_config_data(self):
        config_file = '42_TLM_CONFIG.txt'
        parsed_configs = self.P.parse_config_data(self.tlmConfigFilepath + config_file, True)
        
        self.assertEquals(parsed_configs['subsystem_assignments']['42_TLM.txt'], [['CDH'], ['GNC'], ['GNC'], ['POWER'], ['POWER'], ['POWER'], ['THERMAL'], ['THERMAL'], ['THERMAL'], ['GNC'], ['GNC'], ['GNC'], ['GNC']])
        self.assertEquals(parsed_configs['test_assignments']['42_TLM.txt'], [[['SYNC', 'TIME']], [['FEASIBILITY', -1.0, 0.0, 10.0, 15.0]], [['NOOP']], [['FEASIBILITY', -1.0, 0.0, 10.0, 15.0]], [['NOOP']], [['NOOP']], [['FEASIBILITY', -1.0, 0.0, 10.0, 15.0]], [['NOOP']], [['NOOP']], [['FEASIBILITY', -1.0, 0.0, 10.0, 15.0]], [['NOOP']], [['NOOP']], [['NOOP']]])
        self.assertEquals(parsed_configs['description_assignments']['42_TLM.txt'], ['No description', 'No description', 'No description', 'No description', 'No description', 'No description', 'No description', 'No description', 'No description', 'No description', 'No description', 'No description', 'No description'])

        parsed_configs = self.P.parse_config_data(self.tlmConfigFilepath + config_file, False)
        self.assertEquals(parsed_configs['subsystem_assignments']['42_TLM.txt'], [['MISSION'], ['MISSION'], ['MISSION'], ['MISSION'], ['MISSION'], ['MISSION'], ['MISSION'], ['MISSION'], ['MISSION'], ['MISSION'], ['MISSION'], ['MISSION'], ['MISSION']])

    def test_time_ordering_not_occurring(self):
        dataFiles = 'time_ordering.txt'
        self.P.raw_data_file_path = self.rawDataFilepath
        hdrs, data = self.P.parse_sim_data(dataFiles) # Can only parse one datafile!  

        # Test 'hdrs'
        parsed_filenames = list(hdrs.keys())                       # Just one file parsed
        self.assertEquals(len(parsed_filenames), 1)                # .
        self.assertEquals(parsed_filenames[0], 'time_ordering.txt')       # ..
        self.assertEquals(len(hdrs['time_ordering.txt']), 13)             # 13 TLM points (data fields) 

        # Test 'data'
        self.assertEquals(len(data.values()), 5)                   # 5 time steps 
        for frame in data.values():                                # Parsing for timesteps formatted correctly
            self.assertEquals(list(frame.keys())[0], 'time_ordering.txt') # 13 tlm points in each frame 
            self.assertEquals(len(frame['time_ordering.txt']), 13)        # .

        self.assertNotEquals(list(data.keys()), ['2019-127-12:00:17.300006746', 
                                                 '2019-127-12:00:17.300006747', 
                                                 '2019-127-12:00:17.300006748', 
                                                 '2019-127-12:00:17.300006790', 
                                                 '2019-127-12:00:17.300006801'])
        

if __name__ == '__main__':
    unittest.main()
