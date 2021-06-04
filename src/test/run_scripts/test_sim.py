""" Test Simulator Functionality """ 
import os 
# import sys 
import unittest 

from src.data_handling.data_source import DataSource

from src.run_scripts.sim import Simulator

class TestSimulator(unittest.TestCase):

    def setUp(self):
        self.test_path = os.path.dirname(os.path.abspath(__file__))

    def test_init_sim(self):
        sim_type = 'FortyTwo'
        # parsed_data = TimeSynchronizer()
        SBN_flag = False
        
        # S = Simulator(sim_type, parsed_data, SBN_flag)
        return

if __name__ == '__main__':
    unittest.main()
