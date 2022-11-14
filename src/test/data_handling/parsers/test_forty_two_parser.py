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

    def test_init_empty_parser(self):
        self.assertEqual(self.P.raw_data_file_path, '')
        self.assertEqual(self.P.metadata_file_path, '')
        self.assertEqual(self.P.all_headers, '')
        self.assertEqual(self.P.sim_data, '')
        self.assertEqual(self.P.binning_configs, '')

    def test_init_nonempty_parser(self):
        P = FortyTwo(self.run_path + '/data/raw_telemetry_data/',
                          self.run_path + '/data/telemetry_configs/',
                          str(['42_TLM.txt']),
                          str(['42_TLM_CONFIG.txt']))

        self.assertEqual(P.raw_data_file_path, self.run_path + '/data/raw_telemetry_data/')
        self.assertEqual(P.metadata_file_path, self.run_path + '/data/telemetry_configs/')
        
        self.assertEqual(P.all_headers, {'42_TLM.txt': ['TIME', 
                                                         'SAMPLE.sample_data_tlm_t.sample_data_counter', 
                                                         'SAMPLE.sample_data_tlm_t.sample_data_value', 
                                                         'SAMPLE.sample_data_power_t.sample_data_counter', 
                                                         'SAMPLE.sample_data_power_t.sample_data_voltage', 
                                                         'SAMPLE.sample_data_power_t.sample_data_current', 
                                                         'SAMPLE.sample_data_thermal_t.sample_data_counter', 
                                                         'SAMPLE.sample_data_thermal_t.sample_data_internal_temp', 
                                                         'SAMPLE.sample_data_thermal_t.sample_data_external_temp', 
                                                         'SAMPLE.sample_data_gps_t.sample_data_counter', 
                                                         'SAMPLE.sample_data_gps_t.sample_data_lat', 
                                                         'SAMPLE.sample_data_gps_t.sample_data_lng', 
                                                         'SAMPLE.sample_data_gps_t.sample_data_alt']})
        self.assertEqual(P.sim_data, {'2019-127-12:00:17.300006746': {'42_TLM.txt': ['2019-127-12:00:17.300006746', 
                                                                                      '0', 
                                                                                      '0.000000000000e+00', 
                                                                                      '0', 
                                                                                      '0.000000000000e+00', 
                                                                                      '0.000000000000e+00', 
                                                                                      '0', 
                                                                                      '0.000000000000e+00', 
                                                                                      '0.000000000000e+00', 
                                                                                      '0', 
                                                                                      '0.000000000000e+00', 
                                                                                      '0.000000000000e+00', 
                                                                                      '0.000000000000e+00']}, 
                                       '2019-127-12:00:17.300006747': {'42_TLM.txt': ['2019-127-12:00:17.300006747', 
                                                                                      '1', 
                                                                                      '0.000000000000e+00', 
                                                                                      '0', 
                                                                                      '0.000000000000e+00', 
                                                                                      '0.000000000000e+00', 
                                                                                      '0', 
                                                                                      '0.000000000000e+00', 
                                                                                      '0.000000000000e+00', 
                                                                                      '0', 
                                                                                      '0.000000000000e+00', 
                                                                                      '0.000000000000e+00', 
                                                                                      '0.000000000000e+00']}, 
                                       '2019-127-12:00:17.300006748': {'42_TLM.txt': ['2019-127-12:00:17.300006748', 
                                                                                      '2', 
                                                                                      '0.000000000000e+00', 
                                                                                      '0', 
                                                                                      '0.000000000000e+00', 
                                                                                      '0.000000000000e+00', 
                                                                                      '0', 
                                                                                      '0.000000000000e+00', 
                                                                                      '0.000000000000e+00', 
                                                                                      '0', 
                                                                                      '0.000000000000e+00', 
                                                                                      '0.000000000000e+00', 
                                                                                      '0.000000000000e+00']}, 
                                       '2019-127-12:00:17.300006790': {'42_TLM.txt': ['2019-127-12:00:17.300006790', 
                                                                                      '3', 
                                                                                      '0.000000000000e+00', 
                                                                                      '0', 
                                                                                      '0.000000000000e+00', 
                                                                                      '0.000000000000e+00', 
                                                                                      '0', 
                                                                                      '0.000000000000e+00', 
                                                                                      '0.000000000000e+00', 
                                                                                      '0', 
                                                                                      '0.000000000000e+00', 
                                                                                      '0.000000000000e+00', 
                                                                                      '0.000000000000e+00']},
                                       '2019-127-12:00:17.300006801': {'42_TLM.txt': ['2019-127-12:00:17.300006801', 
                                                                                      '4', 
                                                                                      '0.000000000000e+00', 
                                                                                      '0', 
                                                                                      '0.000000000000e+00', 
                                                                                      '0.000000000000e+00', 
                                                                                      '0', 
                                                                                      '0.000000000000e+00', 
                                                                                      '0.000000000000e+00', 
                                                                                      '0', 
                                                                                      '0.000000000000e+00', 
                                                                                      '0.000000000000e+00', 
                                                                                      '0.000000000000e+00']}})
        self.assertEqual(P.binning_configs, {'subsystem_assignments': {'42_TLM.txt': [['MISSION'], 
                                                                                       ['MISSION'], 
                                                                                       ['MISSION'], 
                                                                                       ['MISSION'], 
                                                                                       ['MISSION'], 
                                                                                       ['MISSION'], 
                                                                                       ['MISSION'], 
                                                                                       ['MISSION'], 
                                                                                       ['MISSION'], 
                                                                                       ['MISSION'], 
                                                                                       ['MISSION'], 
                                                                                       ['MISSION'], 
                                                                                       ['MISSION']]}, 
                                              'test_assignments': {'42_TLM.txt': [[['SYNC', 'TIME']], 
                                                                                  [['FEASIBILITY', -1.0, 0.0, 10.0, 15.0]], 
                                                                                  [['NOOP']], 
                                                                                  [['FEASIBILITY', -1.0, 0.0, 10.0, 15.0]], 
                                                                                  [['NOOP']], 
                                                                                  [['NOOP']], 
                                                                                  [['FEASIBILITY', -1.0, 0.0, 10.0, 15.0]], 
                                                                                  [['NOOP']], 
                                                                                  [['NOOP']], 
                                                                                  [['FEASIBILITY', -1.0, 0.0, 10.0, 15.0]],
                                                                                  [['NOOP']], 
                                                                                  [['NOOP']], 
                                                                                  [['NOOP']]]}, 
                                              'description_assignments': {'42_TLM.txt': ['No description', 
                                                                                         'No description', 
                                                                                         'No description', 
                                                                                         'No description', 
                                                                                         'No description', 
                                                                                         'No description', 
                                                                                         'No description', 
                                                                                         'No description', 
                                                                                         'No description', 
                                                                                         'No description', 
                                                                                         'No description', 
                                                                                         'No description', 
                                                                                         'No description']}})

    def test_parse_sim_data(self):
        dataFiles = '42_TLM.txt'
        self.P.raw_data_file_path = self.rawDataFilepath
        hdrs, data = self.P.parse_sim_data(dataFiles) # Can only parse one datafile!  

        # Test 'hdrs'
        parsed_filenames = list(hdrs.keys())                       # Just one file parsed
        self.assertEqual(len(parsed_filenames), 1)                # .
        self.assertEqual(parsed_filenames[0], '42_TLM.txt')       # ..
        self.assertEqual(len(hdrs['42_TLM.txt']), 13)             # 13 TLM points (data fields) 

        # Test 'data'
        self.assertEqual(len(data.values()), 5)                   # 5 time steps 
        for frame in data.values():                                # Parsing for timesteps formatted correctly
            self.assertEqual(list(frame.keys())[0], '42_TLM.txt') # 13 tlm points in each frame 
            self.assertEqual(len(frame['42_TLM.txt']), 13)        # .

        # Test time ordering
        self.assertEqual(list(data.keys()), ['2019-127-12:00:17.300006746', 
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
        self.assertEqual(len(hdrs), 13)
        self.assertEqual(hdrs, ['TIME', 'SAMPLE.sample_data_tlm_t.sample_data_counter', 'SAMPLE.sample_data_tlm_t.sample_data_value', 'SAMPLE.sample_data_power_t.sample_data_counter', 'SAMPLE.sample_data_power_t.sample_data_voltage', 'SAMPLE.sample_data_power_t.sample_data_current', 'SAMPLE.sample_data_thermal_t.sample_data_counter', 'SAMPLE.sample_data_thermal_t.sample_data_internal_temp', 'SAMPLE.sample_data_thermal_t.sample_data_external_temp', 'SAMPLE.sample_data_gps_t.sample_data_counter', 'SAMPLE.sample_data_gps_t.sample_data_lat', 'SAMPLE.sample_data_gps_t.sample_data_lng', 'SAMPLE.sample_data_gps_t.sample_data_alt'])

    def test_parse_frame(self):
        txt_file = open(self.rawDataFilepath + 'single_frame.txt',"r+")
        data_str = txt_file.read().split('\n[EOF]')[0]
        txt_file.close()
        clean_frame = self.P.parse_frame(data_str) 
        self.assertEqual(clean_frame, ['2019-127-12:00:17.300006801', '4', '0.000000000000e+00', '0', '0.000000000000e+00', '0.000000000000e+00', '0', '0.000000000000e+00', '0.000000000000e+00', '0', '0.000000000000e+00', '0.000000000000e+00', '0.000000000000e+00'])

    def test_parse_config_data(self):
        config_file = '42_TLM_CONFIG.txt'
        parsed_configs = self.P.parse_config_data(self.tlmConfigFilepath + config_file, True)
        
        self.assertEqual(parsed_configs['subsystem_assignments']['42_TLM.txt'], [['CDH'], ['GNC'], ['GNC'], ['POWER'], ['POWER'], ['POWER'], ['THERMAL'], ['THERMAL'], ['THERMAL'], ['GNC'], ['GNC'], ['GNC'], ['GNC']])
        self.assertEqual(parsed_configs['test_assignments']['42_TLM.txt'], [[['SYNC', 'TIME']], [['FEASIBILITY', -1.0, 0.0, 10.0, 15.0]], [['NOOP']], [['FEASIBILITY', -1.0, 0.0, 10.0, 15.0]], [['NOOP']], [['NOOP']], [['FEASIBILITY', -1.0, 0.0, 10.0, 15.0]], [['NOOP']], [['NOOP']], [['FEASIBILITY', -1.0, 0.0, 10.0, 15.0]], [['NOOP']], [['NOOP']], [['NOOP']]])
        self.assertEqual(parsed_configs['description_assignments']['42_TLM.txt'], ['No description', 'No description', 'No description', 'No description', 'No description', 'No description', 'No description', 'No description', 'No description', 'No description', 'No description', 'No description', 'No description'])

        parsed_configs = self.P.parse_config_data(self.tlmConfigFilepath + config_file, False)
        self.assertEqual(parsed_configs['subsystem_assignments']['42_TLM.txt'], [['MISSION'], ['MISSION'], ['MISSION'], ['MISSION'], ['MISSION'], ['MISSION'], ['MISSION'], ['MISSION'], ['MISSION'], ['MISSION'], ['MISSION'], ['MISSION'], ['MISSION']])

    def test_time_ordering_not_occurring(self):
        dataFiles = 'time_ordering.txt'
        self.P.raw_data_file_path = self.rawDataFilepath
        hdrs, data = self.P.parse_sim_data(dataFiles) # Can only parse one datafile!  

        # Test 'hdrs'
        parsed_filenames = list(hdrs.keys())                       # Just one file parsed
        self.assertEqual(len(parsed_filenames), 1)                # .
        self.assertEqual(parsed_filenames[0], 'time_ordering.txt')       # ..
        self.assertEqual(len(hdrs['time_ordering.txt']), 13)             # 13 TLM points (data fields) 

        # Test 'data'
        self.assertEqual(len(data.values()), 5)                   # 5 time steps 
        for frame in data.values():                                # Parsing for timesteps formatted correctly
            self.assertEqual(list(frame.keys())[0], 'time_ordering.txt') # 13 tlm points in each frame 
            self.assertEqual(len(frame['time_ordering.txt']), 13)        # .

        self.assertNotEqual(list(data.keys()), ['2019-127-12:00:17.300006746', 
                                                 '2019-127-12:00:17.300006747', 
                                                 '2019-127-12:00:17.300006748', 
                                                 '2019-127-12:00:17.300006790', 
                                                 '2019-127-12:00:17.300006801'])
        
    def test_get_sim_data(self):
        hdrs, data, configs = self.P.get_sim_data()
        self.assertEqual(hdrs, '')
        self.assertEqual(data, '')
        self.assertEqual(configs, '')


if __name__ == '__main__':
    unittest.main()
