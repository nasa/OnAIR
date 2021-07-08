  
""" Test Status Functionality """
import os
import unittest

from src.systems.status import Status
from src.systems.telemetry_test_suite import TelemetryTestSuite

class TestTelemetryTestSuite(unittest.TestCase):

    def setUp(self):
        self.test_path = os.path.dirname(os.path.abspath(__file__))
        self.TTS = TelemetryTestSuite(['TIME', 'A', 'B'], [[['SYNC', 'TIME']], [['NOOP']], [['NOOP']]])

    def test_init_empty_testsuite(self):
        TTS = TelemetryTestSuite()
        self.assertEquals(TTS.dataFields, [])
        self.assertEquals(TTS.tests, [])
        self.assertEquals(TTS.epsilon, 0.00001)
        self.assertEquals(TTS.latest_results, None)

    def test_init_nonempty_testsuite(self):
        self.assertEquals(self.TTS.dataFields, ['TIME', 'A', 'B'])
        self.assertEquals(self.TTS.tests, [[['SYNC', 'TIME']], [['NOOP']], [['NOOP']]])
        self.assertEquals(self.TTS.epsilon, 0.00001)
        self.assertEquals(self.TTS.latest_results, None)

    def test_execute(self):
        frame = [1, 2, 3]
        self.assertEquals(self.TTS.latest_results, None)
        self.TTS.execute_suite(frame)
        self.assertEquals(len(self.TTS.latest_results), 3)

    def test_run_tests(self):
        i = 0
        val = 1
        sync_data = {}
        result = self.TTS.run_tests(i, val, sync_data)

        self.assertEquals(type(result), Status)

    def test_get_latest_result(self):
        self.assertEquals(self.TTS.get_latest_result('TIME'), None)
        self.assertEquals(self.TTS.get_latest_result('A'), None)
        self.assertEquals(self.TTS.get_latest_result('B'), None)

        self.TTS.execute_suite([1, 2, 3])
        self.assertEquals(type(self.TTS.get_latest_result('TIME')), Status)
        self.assertEquals(type(self.TTS.get_latest_result('A')), Status)
        self.assertEquals(type(self.TTS.get_latest_result('B')), Status)

    def test_sync(self):
        val = 1
        params = [1] # this is fed to the test suite... Should re-implement
        epsilon = 0.000001

        result = self.TTS.sync(val, params, epsilon)
        self.assertEquals(result[0], 'GREEN')
        self.assertEquals(result[1], [({'GREEN'}, 1.0)]) # Rethink this

    def test_rotational(self):
        val = 1
        params = [1] 
        epsilon = 0.000001

        result = self.TTS.rotational(val, params, epsilon)

        self.assertEquals(result[0], 'YELLOW')
        self.assertEquals(result[1], []) 

    def test_state(self):
                # Gr,  Ylw,   Rd
        params = [[1], [0,2], [3]]
        epsilon = 0.000001
                              # val
        result = self.TTS.state(1, params, epsilon)
        self.assertEquals(result[0], 'GREEN')
        self.assertEquals(result[1], [({'GREEN'}, 1.0)]) 

        result = self.TTS.state(0, params, epsilon)
        self.assertEquals(result[0], 'YELLOW')
        self.assertEquals(result[1], [({'YELLOW'}, 1.0)]) 

        result = self.TTS.state(3, params, epsilon)
        self.assertEquals(result[0], 'RED')
        self.assertEquals(result[1], [({'RED'}, 1.0)]) 


        result = self.TTS.state(4, params, epsilon)
        self.assertEquals(result[0], '---')
        self.assertEquals(result[1], [({'GREEN', 'RED', 'YELLOW'}, 1.0)]) 

    def test_feasibility(self):
        epsilon = 0.000001
        #Test with param length of 2
        params = [0, 10]        
        #Test on lower boundary
        val = 0
        state, mass_assignments = self.TTS.feasibility(val, params, epsilon)
        self.assertEquals(state, 'RED')
        self.assertEquals(mass_assignments, [({'left_stat' : 'RED', 'stat': 'RED', 'right_stat':'GREEN'}, 1.0)])
        #Test one above lower boundary
        val = 1
        state, mass_assignments = self.TTS.feasibility(val, params, epsilon)
        self.assertEquals(state, 'GREEN')
        self.assertEquals(mass_assignments, [({'left_stat' : '', 'stat': 'GREEN', 'right_stat':''}, 1.0)])
        #Test on upper boundary
        val = 10
        state, mass_assignments = self.TTS.feasibility(val, params, epsilon)
        self.assertEquals(state, 'RED')
        self.assertEquals(mass_assignments, [({'left_stat' : 'GREEN', 'stat': 'RED', 'right_stat':'RED'}, 1.0)])
        #Test one below upper boundary
        val = 9
        state, mass_assignments = self.TTS.feasibility(val, params, epsilon)
        self.assertEquals(state, 'GREEN')
        self.assertEquals(mass_assignments, [({'left_stat' : '', 'stat': 'GREEN', 'right_stat':''}, 1.0)])
        #Test in middle of boundaries
        val = 5
        state, mass_assignments = self.TTS.feasibility(val, params, epsilon)
        self.assertEquals(state, 'GREEN')
        self.assertEquals(mass_assignments, [({'left_stat' : '', 'stat': 'GREEN', 'right_stat':''}, 1.0)]) 
        #Test below lower boundary
        val = -5
        state, mass_assignments = self.TTS.feasibility(val, params, epsilon)
        self.assertEquals(state, 'RED')
        self.assertEquals(mass_assignments, [({'left_stat' : 'RED', 'stat': 'RED', 'right_stat':''}, 1.0)]) 
        #Test above upper boundary
        val = 15
        state, mass_assignments = self.TTS.feasibility(val, params, epsilon)
        self.assertEquals(state, 'RED')
        self.assertEquals(mass_assignments, [({'left_stat' : '', 'stat': 'RED', 'right_stat':'RED'}, 1.0)])
        #Test with param length of 4
        params = [0,10,20,30]
        #Test in lower yellow range        
        val = 5
        state, mass_assignments = self.TTS.feasibility(val, params, epsilon)
        self.assertEquals(state, 'YELLOW')
        self.assertEquals(mass_assignments, [({'left_stat' : '', 'stat': 'YELLOW', 'right_stat':''}, 1.0)])
        #Test in green range
        val = 15
        state, mass_assignments = self.TTS.feasibility(val, params, epsilon)
        self.assertEquals(state, 'GREEN')
        self.assertEquals(mass_assignments, [({'left_stat' : '', 'stat': 'GREEN', 'right_stat':''}, 1.0)])
        #Test in higher yellow range
        val = 25
        state, mass_assignments = self.TTS.feasibility(val, params, epsilon)
        self.assertEquals(state, 'YELLOW')
        self.assertEquals(mass_assignments, [({'left_stat' : '', 'stat': 'YELLOW', 'right_stat':''}, 1.0)]) 
        return

    def test_noop(self):
        result = self.TTS.noop(1, [], 0.001)
        self.assertEquals(result[0], 'GREEN')
        self.assertEquals(result[1], [({'GREEN'}, 1.0)]) 

    def test_calc_single_status(self):
        status_list = ['RED', 'RED', 'GREEN', 'YELLOW', 'GREEN', 'GREEN']
        result, confidence = self.TTS.calc_single_status(status_list, mode='strict')
        self.assertEquals(result, 'RED')
        self.assertEquals(confidence, 0.3333333333333333)
        result, confidence = self.TTS.calc_single_status(status_list, mode='distr')
        self.assertEquals(result, 'GREEN')
        self.assertEquals(confidence, 0.5)
        result, confidence = self.TTS.calc_single_status(status_list, mode='max')
        self.assertEquals(result, 'GREEN')
        self.assertEquals(confidence, 1.0)

        status_list = ['RED', 'RED', 'RED', 'YELLOW', 'GREEN', 'GREEN']
        result, confidence = self.TTS.calc_single_status(status_list, mode='strict')
        self.assertEquals(result, 'RED')
        self.assertEquals(confidence, 0.5)
        result, confidence = self.TTS.calc_single_status(status_list, mode='distr')
        self.assertEquals(result, 'RED')
        self.assertEquals(confidence, 0.5)
        result, confidence = self.TTS.calc_single_status(status_list, mode='max')
        self.assertEquals(result, 'RED')
        self.assertEquals(confidence, 1.0)

        status_list = ['YELLOW', 'GREEN', 'YELLOW', 'YELLOW', 'YELLOW', 'GREEN']
        result, confidence = self.TTS.calc_single_status(status_list, mode='strict')
        self.assertEquals(result, 'YELLOW')
        self.assertEquals(confidence, 1.0)
        result, confidence = self.TTS.calc_single_status(status_list, mode='distr')
        self.assertEquals(result, 'YELLOW')
        self.assertEquals(confidence, 0.6666666666666666)
        result, confidence = self.TTS.calc_single_status(status_list, mode='max')
        self.assertEquals(result, 'YELLOW')
        self.assertEquals(confidence, 1.0)
        return


if __name__ == '__main__':
    unittest.main()
