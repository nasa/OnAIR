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

        headers = {'sample_1' : ['TIME', 'A', 'B']}      
        dataFrames = {'1' : {'sample_1' : ['1035','202','0.3']},
                      '2' : {'sample_1' : ['1005','202','0.3']},
                      '3' : {'sample_1' : ['1350','202','0.3']}}
        test_configs = {'test_assignments': {'sample_1': [[['SYNC', 'TIME']], [['NOOP']], [['NOOP']]]}}
        self.TS = TimeSynchronizer(headers,dataFrames,test_configs)
    
    def test_init(self):
        sim_type = 'FortyTwo'

        parsed_data = self.TS
            
        SBN_flag = False
        
        S = Simulator(sim_type, parsed_data, SBN_flag)

        self.assertTrue(S.simulator, 'FortyTwo')
        self.assertTrue(type(S.simData), DataSource)
        self.assertTrue(type(S.brain), Brain)

    def test_run_sim(self):
        parsed_data = self.TS
        S = Simulator('FortyTwo', parsed_data, False)

        IO_flag = False
        dev_flag = False
        viz_flag = False

        S.run_sim(IO_flag, dev_flag, viz_flag)



if __name__ == '__main__':
    unittest.main()
