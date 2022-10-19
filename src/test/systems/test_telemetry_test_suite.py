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
        params = [1] # this is fed to the tset suite... hmmm ... I should prob do this diff
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
        return

    def test_noop(self):
        result = self.TTS.noop(1, [], 0.001)
        self.assertEquals(result[0], 'GREEN')
        self.assertEquals(result[1], [({'GREEN'}, 1.0)]) 

    def test_calc_single_status(self):
        return


if __name__ == '__main__':
    unittest.main()
