""" Test Binner Functionality """
import os
# import sys
import unittest
# import shutil

# from src.run_scripts.execution_engine import ExecutionEngine
from src.data_handling.binner import Binner

class TestBinner(unittest.TestCase):

    def setUp(self):
        self.test_path = os.path.dirname(os.path.abspath(__file__))
        self.B = Binner()

    def test_init_empty_binner(self):
        self.assertEquals(self.B.headers, {})
        self.assertEquals(self.B.sources, [])
        # self.assertEquals(self.B.ss_assigns, {})
        self.assertEquals(self.B.test_assigns, {})
        self.assertEquals(self.B.desc_assigns, {})

        # self.assertEquals(self.B.sim_data, {})
        # self.assertEquals(self.B.sc_configs, {})

    # def test_init_data(self):
    #     hdrs = {'test_sample' : ['TIME', 'hdr_A', 'hdr_B']}
    #     data = {'1234' : {'test_sample' : ['1234','202','0.3']}}
        
    #     # Even if you give configs with ss assignments, they should not be here at the binner stage 
    #     configs = {'subsystem_assignments': {'test_sample': [['MISSION'], ['MISSION'], ['MISSION']]}, 
    #                'test_assignments': {'test_sample': [[['SYNC', 'TIME']], [['NOOP']], [['NOOP']]]}, 
    #                'description_assignments': {'test_sample': ['Time', 'No description', 'No description']}}

    #     self.B.init_data(hdrs, data, configs) 

    #     self.assertEquals(self.B.headers, {'test_sample' : ['TIME', 'hdr_A', 'hdr_B']})
    #     self.assertEquals(self.B.sources, ['test_sample'])
    #     # self.assertEquals(self.B.ss_assigns, {'test_sample': [['MISSION'], ['MISSION'], ['MISSION']]})
    #     self.assertEquals(self.B.test_assigns, {'test_sample': [[['SYNC', 'TIME']], [['NOOP']], [['NOOP']]]})
    #     self.assertEquals(self.B.desc_assigns, {'test_sample': ['Time', 'No description', 'No description']})

        # self.sim_data = binned_data
        # self.sc_configs = metadata



        return

if __name__ == '__main__':
    unittest.main()
