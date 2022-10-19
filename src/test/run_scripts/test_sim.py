""" Test Simulator Functionality """ 
import os 
# import sys 
import unittest 

from src.data_handling.data_source import DataSource
from src.reasoning.brain import Brain
from src.data_handling.time_synchronizer import TimeSynchronizer
from src.run_scripts.sim import Simulator

class TestSimulator(unittest.TestCase):

    def setUp(self):
        self.test_path = os.path.dirname(os.path.abspath(__file__))

    def test_init_sim(self):
        sim_type = 'FortyTwo'
        parsed_data = TimeSynchronizer()
        parsed_data.ordered_fused_headers = ['TIME', 'A', 'B']
        parsed_data.ordered_fused_tests = [[['SYNC', 'TIME']], [['NOOP']], [['NOOP']]]
        parsed_data.sim_data = [[1,1,1],[2,2,2],[3,3,3]]
        SBN_flag = False
        
        S = Simulator(sim_type, parsed_data, SBN_flag)

        self.assertTrue(S.simulator, 'FortyTwo')
        self.assertTrue(type(S.simData), DataSource)
        self.assertTrue(type(S.brain), Brain)

    def test_run_sim(self):
        parsed_data = TimeSynchronizer()
        parsed_data.ordered_fused_headers = ['TIME', 'A', 'B']
        parsed_data.ordered_fused_tests = [[['SYNC', 'TIME']], [['NOOP']], [['NOOP']]]
        parsed_data.sim_data = [[1,1,1],[2,2,2],[3,3,3]]
        
        S = Simulator('FortyTwo', parsed_data, False)

        IO_flag = False
        dev_flag = False
        viz_flag = False

        S.run_sim(IO_flag, dev_flag, viz_flag)



if __name__ == '__main__':
    unittest.main()
