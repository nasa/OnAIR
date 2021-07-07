""" Test Time Sync Functionality """
import os
import unittest

from src.data_handling.time_synchronizer import TimeSynchronizer

class TestTimeSynchronizer(unittest.TestCase):

    def setUp(self):
        self.test_path = os.path.dirname(os.path.abspath(__file__))
        headers = {'test_sample_01' : ['TIME', 'hdr_A', 'hdr_B'],
                   'test_sample_02' : ['TIME', 'hdr_C']}      
        dataFrames = {'1234' : {'test_sample_01' : ['1234','202','0.3'],
                                'test_sample_02' : ['1234','0.3']},
                      '2235' : {'test_sample_02' : ['2235','202']},
                      '1035' : {'test_sample_01' : ['1035','202','0.3'],
                                'test_sample_02' : ['1035','0.3']},
                      '1305' : {'test_sample_01' : ['1005','202','0.3']},
                      '1350' : {'test_sample_01' : ['1350','202','0.3'],
                                'test_sample_02' : ['1350','0.3']}}
        test_configs = {'test_assignments': {'test_sample_01': [[['SYNC', 'TIME']], [['NOOP']], [['NOOP']]],
                                             'test_sample_02': [[['SYNC', 'TIME']], [['NOOP']]]}, 
                        'description_assignments': {'test_sample_01': ['Time', 'No description', 'No description']}}
 
        self.TS = TimeSynchronizer(headers,dataFrames,test_configs)

    def test_init(self):
        headers = {'test_sample_01' : ['TIME', 'hdr_A', 'hdr_B'],
                   'test_sample_02' : ['TIME', 'hdr_C']}      
        dataFrames = {'1234' : {'test_sample_01' : ['1234','202','0.3'],
                                'test_sample_02' : ['1234','0.3']},
                      '2235' : {'test_sample_02' : ['2235','202']},
                      '1035' : {'test_sample_01' : ['1035','202','0.3'],
                                'test_sample_02' : ['1035','0.3']},
                      '1305' : {'test_sample_01' : ['1005','202','0.3']},
                      '1350' : {'test_sample_01' : ['1350','202','0.3'],
                                'test_sample_02' : ['1350','0.3']}}
        test_configs = {'test_assignments': {'test_sample_01': [[['SYNC', 'TIME']], [['NOOP']], [['NOOP']]],
                                             'test_sample_02': [[['SYNC', 'TIME']], [['NOOP']]]}, 
                        'description_assignments': {'test_sample_01': ['Time', 'No description', 'No description']}}
 
        TS = TimeSynchronizer(headers,dataFrames,test_configs)

        self.assertEquals(self.TS.ordered_fused_tests, [[['SYNC', 'TIME']], [['NOOP']], [['NOOP']], [['NOOP']]])
        self.assertEquals(self.TS.ordered_sources, ['test_sample_01', 'test_sample_02'])
        self.assertEquals(self.TS.ordered_fused_headers, ['TIME', 'hdr_A', 'hdr_B', 'hdr_C'])
        self.assertEquals(self.TS.indices_to_remove, [0,3])
        self.assertEquals(self.TS.offsets, {'test_sample_01': 0, 'test_sample_02': 3})
        self.assertEquals(TS.sim_data, [['1035', '202', '0.3', '0.3'], 
                                        ['1234', '202', '0.3', '0.3'], 
                                        ['1305', '202', '0.3', '-'], 
                                        ['1350', '202', '0.3', '0.3'], 
                                        ['2235', '-', '-', '202']])

    def test_init_sync_data(self):
        hdrs = {'test_sample_01' : ['TIME', 'hdr_1', 'hdr_2'],
                'test_sample_02' : ['TIME', 'hdr_3']}        
        
        # Even if you give configs with ss assignments, they should not be here at the binner stage 
        configs = {'test_assignments': {'test_sample_01': [[['SYNC', 'TIME']], [['NOOP']], [['NOOP']]],
                                        'test_sample_02': [[['SYNC', 'TIME']], [['NOOP']]]}, 
                   'description_assignments': {'test_sample_01': ['Time', 'No description', 'No description']}}

        self.TS.init_sync_data(hdrs, configs) 

        self.assertEquals(self.TS.ordered_fused_tests, [[['SYNC', 'TIME']], [['NOOP']], [['NOOP']], [['NOOP']]])
        self.assertEquals(self.TS.ordered_sources, ['test_sample_01', 'test_sample_02'])
        self.assertEquals(self.TS.ordered_fused_headers, ['TIME', 'hdr_1', 'hdr_2', 'hdr_3'])
        self.assertEquals(self.TS.indices_to_remove, [0,3])
        self.assertEquals(self.TS.offsets, {'test_sample_01': 0, 'test_sample_02': 3})

    def test_sort_data(self):

        self.TS.ordered_fused_tests = [[['SYNC', 'TIME']], [['NOOP']], [['NOOP']], [['NOOP']]]
        self.TS.ordered_sources = ['test_sample_01', 'test_sample_02']
        self.TS.ordered_fused_headers = ['TIME', 'hdr_A', 'hdr_B', 'hdr_C']
        self.TS.indices_to_remove =[0,3]
        self.TS.offsets = {'test_sample_01': 0, 'test_sample_02': 3}
        self.TS.unclean_fused_hdrs = ['TIME', 'hdr_A', 'hdr_B', 'TIME', 'hdr_C']

        data = {'1234' : {'test_sample_01' : ['1234','202','0.3'],
                          'test_sample_02' : ['1234','0.3']},
                '2235' : {'test_sample_02' : ['2235','202']},
                '1035' : {'test_sample_01' : ['1035','202','0.3'],
                          'test_sample_02' : ['1035','0.3']},
                '1305' : {'test_sample_01' : ['1005','202','0.3']},
                '1350' : {'test_sample_01' : ['1350','202','0.3'],
                          'test_sample_02' : ['1350','0.3']}}

        self.TS.sort_data(data)

        self.assertEquals(self.TS.sim_data, [['1035', '202', '0.3', '0.3'], 
                                             ['1234', '202', '0.3', '0.3'], 
                                             ['1305', '202', '0.3', '-'], 
                                             ['1350', '202', '0.3', '0.3'], 
                                             ['2235', '-', '-', '202']])

    def test_remove_time_headers(self):
        hdrs_list = ['A', 'B', 'time', 'TIME', 'C', 'D']
        indices, clean_hdrs_list = self.TS.remove_time_headers(hdrs_list)
        self.assertEquals(clean_hdrs_list, ['A', 'B','C', 'D'])

    def test_remove_time_datapoints(self):
        data = ['1', '2', '3', '4']
        indices_to_remove = [0, 2]
        clean_data = self.TS.remove_time_datapoints(data, indices_to_remove)
        self.assertEquals(clean_data, ['2', '4'])

        data = [['list1'], [], ['list3', 'list3'], []]
        indices_to_remove = [0, 2]
        clean_data = self.TS.remove_time_datapoints(data, indices_to_remove)
        self.assertEquals(clean_data, [[],[]])

    def test_get_spacecraft_metadata(self):
        return 

    def test_get_sim_data(self):
        return
        
if __name__ == '__main__':
    unittest.main()
