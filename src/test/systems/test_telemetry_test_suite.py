  
""" Test Status Functionality """
import pytest
from mock import MagicMock

from src.systems.status import Status
from src.systems.telemetry_test_suite import TelemetryTestSuite


# __init__ tests
def test__init__sets_the_expected_values_with_given_headers_and_tests(mocker):
    # Arrange
    arg_headers = MagicMock()
    arg_tests = MagicMock()

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    # Act
    cut.__init__(arg_headers, arg_tests)

    # Assert
    assert cut.dataFields == arg_headers
    assert cut.tests == arg_tests
    assert cut.latest_results == None
    assert cut.epsilon == 1/100000 # production codes notes this value as needing intelligent definition
    assert cut.all_tests == {'SYNC' : cut.sync,
                       'ROTATIONAL' : cut.rotational, 
                            'STATE' : cut.state,
                      'FEASIBILITY' : cut.feasibility, 
                             'NOOP' : cut.noop}

def test__init__default_arg_tests_is_empty_list(mocker):
    # Arrange
    arg_headers = MagicMock()

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    # Act
    cut.__init__(arg_headers)

    # Assert
    assert cut.tests == []

def test__init__default_arg_headers_is_empty_list(mocker):
    # Arrange
    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    # Act
    cut.__init__()

    # Assert
    assert cut.dataFields == []

# execute_suite tests
def test_execute_suite_sets_the_latest_results_to_empty_list_when_updated_frame_len_is_0(mocker):
    # Arrange
    arg_update_frame = '' # empty string for len of 0
    arg_sync_data = MagicMock()

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    # Act
    cut.execute_suite(arg_update_frame, arg_sync_data)

    # Assert
    assert cut.latest_results == []

def test_execute_suite_sets_latests_results_to_list_of_run_tests_for_each_item_in_given_updated_frame(mocker):
    # Arrange
    arg_update_frame = []
    arg_sync_data = MagicMock()

    num_items_in_update = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10
    expected_results = []

    for i in range(num_items_in_update):
        arg_update_frame.append(MagicMock())
        expected_results.append(MagicMock())

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    mocker.patch.object(cut, 'run_tests', side_effect=expected_results)

    # Act
    cut.execute_suite(arg_update_frame, arg_sync_data)

    # Assert
    assert cut.run_tests.call_count == num_items_in_update
    for i in range(num_items_in_update):
        assert cut.run_tests.call_args_list[i].args == (i, arg_update_frame[i], arg_sync_data, )
    assert cut.latest_results == expected_results

def test_execute_suite_default_arg_sync_data_is_empty_map(mocker):
    # Arrange
    arg_update_frame = [MagicMock()]

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    mocker.patch.object(cut, 'run_tests', return_value=86) # arbitrary 86

    # Act
    cut.execute_suite(arg_update_frame)

    # Assert
    assert cut.run_tests.call_args_list[0].args == (0, arg_update_frame[0], {})

# class TestTelemetryTestSuite(unittest.TestCase):

#     def setUp(self):
#         self.test_path = os.path.dirname(os.path.abspath(__file__))
#         self.TTS = TelemetryTestSuite(['TIME', 'A', 'B'], [[['SYNC', 'TIME']], [['NOOP']], [['NOOP']]])

#     def test_init_empty_testsuite(self):
#         TTS = TelemetryTestSuite()
#         self.assertEqual(TTS.dataFields, [])
#         self.assertEqual(TTS.tests, [])
#         self.assertEqual(TTS.epsilon, 0.00001)
#         self.assertEqual(TTS.latest_results, None)

#     def test_init_nonempty_testsuite(self):
#         self.assertEqual(self.TTS.dataFields, ['TIME', 'A', 'B'])
#         self.assertEqual(self.TTS.tests, [[['SYNC', 'TIME']], [['NOOP']], [['NOOP']]])
#         self.assertEqual(self.TTS.epsilon, 0.00001)
#         self.assertEqual(self.TTS.latest_results, None)

#     def test_execute(self):
#         frame = [1, 2, 3]
#         self.assertEqual(self.TTS.latest_results, None)
#         self.TTS.execute_suite(frame)
#         self.assertEqual(len(self.TTS.latest_results), 3)

#     def test_run_tests(self):
#         i = 0
#         val = 1
#         sync_data = {}
#         result = self.TTS.run_tests(i, val, sync_data)

#         self.assertEqual(type(result), Status)

#     def test_get_latest_result(self):
#         self.assertEqual(self.TTS.get_latest_result('TIME'), None)
#         self.assertEqual(self.TTS.get_latest_result('A'), None)
#         self.assertEqual(self.TTS.get_latest_result('B'), None)

#         self.TTS.execute_suite([1, 2, 3])
#         self.assertEqual(type(self.TTS.get_latest_result('TIME')), Status)
#         self.assertEqual(type(self.TTS.get_latest_result('A')), Status)
#         self.assertEqual(type(self.TTS.get_latest_result('B')), Status)

#     def test_sync(self):
#         val = 1
#         params = [1] # this is fed to the test suite... Should re-implement
#         epsilon = 0.000001

#         result = self.TTS.sync(val, params, epsilon)
#         self.assertEqual(result[0], 'GREEN')
#         self.assertEqual(result[1], [({'GREEN'}, 1.0)]) # Rethink this

#     def test_rotational(self):
#         val = 1
#         params = [1] 
#         epsilon = 0.000001

#         result = self.TTS.rotational(val, params, epsilon)

#         self.assertEqual(result[0], 'YELLOW')
#         self.assertEqual(result[1], []) 

#     def test_state(self):
#                 # Gr,  Ylw,   Rd
#         params = [[1], [0,2], [3]]
#         epsilon = 0.000001
#                               # val
#         result = self.TTS.state(1, params, epsilon)
#         self.assertEqual(result[0], 'GREEN')
#         self.assertEqual(result[1], [({'GREEN'}, 1.0)]) 

#         result = self.TTS.state(0, params, epsilon)
#         self.assertEqual(result[0], 'YELLOW')
#         self.assertEqual(result[1], [({'YELLOW'}, 1.0)]) 

#         result = self.TTS.state(3, params, epsilon)
#         self.assertEqual(result[0], 'RED')
#         self.assertEqual(result[1], [({'RED'}, 1.0)]) 


#         result = self.TTS.state(4, params, epsilon)
#         self.assertEqual(result[0], '---')
#         self.assertEqual(result[1], [({'GREEN', 'RED', 'YELLOW'}, 1.0)]) 

#     def test_feasibility(self):
#         epsilon = 0.000001
#         #Test with param length of 2
#         params = [0, 10]        
#         #Test on lower boundary
#         val = 0
#         state, mass_assignments = self.TTS.feasibility(val, params, epsilon)
#         self.assertEqual(state, 'RED')
#         self.assertEqual(mass_assignments, [({'RED', 'GREEN'}, 1.0)])
#         #Test one above lower boundary
#         val = 1
#         state, mass_assignments = self.TTS.feasibility(val, params, epsilon)
#         self.assertEqual(state, 'GREEN')
#         self.assertEqual(mass_assignments, [({'GREEN'}, 1.0)])
#         #Test on upper boundary
#         val = 10
#         state, mass_assignments = self.TTS.feasibility(val, params, epsilon)
#         self.assertEqual(state, 'RED')
#         self.assertEqual(mass_assignments, [({'GREEN', 'RED'}, 1.0)])
#         #Test one below upper boundary
#         val = 9
#         state, mass_assignments = self.TTS.feasibility(val, params, epsilon)
#         self.assertEqual(state, 'GREEN')
#         self.assertEqual(mass_assignments, [({'GREEN'}, 1.0)])
#         #Test in middle of boundaries
#         val = 5
#         state, mass_assignments = self.TTS.feasibility(val, params, epsilon)
#         self.assertEqual(state, 'GREEN')
#         self.assertEqual(mass_assignments, [({'GREEN'}, 1.0)]) 
#         #Test below lower boundary
#         val = -5
#         state, mass_assignments = self.TTS.feasibility(val, params, epsilon)
#         self.assertEqual(state, 'RED')
#         self.assertEqual(mass_assignments, [({'RED'}, 1.0)]) 
#         #Test above upper boundary
#         val = 15
#         state, mass_assignments = self.TTS.feasibility(val, params, epsilon)
#         self.assertEqual(state, 'RED')
#         self.assertEqual(mass_assignments, [({'RED'}, 1.0)])
#         #Test with param length of 4
#         params = [0,10,20,30]
#         #Test in lower yellow range        
#         val = 5
#         state, mass_assignments = self.TTS.feasibility(val, params, epsilon)
#         self.assertEqual(state, 'YELLOW')
#         self.assertEqual(mass_assignments, [({'YELLOW'}, 1.0)])
#         #Test in lower green boundary        
#         val = 10
#         state, mass_assignments = self.TTS.feasibility(val, params, epsilon)
#         self.assertEqual(state, 'GREEN')
#         self.assertEqual(mass_assignments, [({'YELLOW', 'GREEN'}, 1.0)])
#         #Test in green range
#         val = 15
#         state, mass_assignments = self.TTS.feasibility(val, params, epsilon)
#         self.assertEqual(state, 'GREEN')
#         self.assertEqual(mass_assignments, [({'GREEN'}, 1.0)])
#         #Test in higher yellow range
#         val = 25
#         state, mass_assignments = self.TTS.feasibility(val, params, epsilon)
#         self.assertEqual(state, 'YELLOW')
#         self.assertEqual(mass_assignments, [({'YELLOW'}, 1.0)])
#         #Test in lower yellow boundary        
#         val = 20
#         state, mass_assignments = self.TTS.feasibility(val, params, epsilon)
#         self.assertEqual(state, 'YELLOW')
#         self.assertEqual(mass_assignments, [({'GREEN','YELLOW'}, 1.0)])
#         return

#     def test_noop(self):
#         result = self.TTS.noop(1, [], 0.001)
#         self.assertEqual(result[0], 'GREEN')
#         self.assertEqual(result[1], [({'GREEN'}, 1.0)]) 

#     def test_calc_single_status(self):
#         status_list = ['RED', 'RED', 'GREEN', 'YELLOW', 'GREEN', 'GREEN']
#         result, confidence = self.TTS.calc_single_status(status_list, mode='strict')
#         self.assertEqual(result, 'RED')
#         self.assertEqual(confidence, 0.3333333333333333)
#         result, confidence = self.TTS.calc_single_status(status_list, mode='distr')
#         self.assertEqual(result, 'GREEN')
#         self.assertEqual(confidence, 0.5)
#         result, confidence = self.TTS.calc_single_status(status_list, mode='max')
#         self.assertEqual(result, 'GREEN')
#         self.assertEqual(confidence, 1.0)

#         status_list = ['RED', 'RED', 'RED', 'YELLOW', 'GREEN', 'GREEN']
#         result, confidence = self.TTS.calc_single_status(status_list, mode='strict')
#         self.assertEqual(result, 'RED')
#         self.assertEqual(confidence, 0.5)
#         result, confidence = self.TTS.calc_single_status(status_list, mode='distr')
#         self.assertEqual(result, 'RED')
#         self.assertEqual(confidence, 0.5)
#         result, confidence = self.TTS.calc_single_status(status_list, mode='max')
#         self.assertEqual(result, 'RED')
#         self.assertEqual(confidence, 1.0)

#         status_list = ['YELLOW', 'GREEN', 'YELLOW', 'YELLOW', 'YELLOW', 'GREEN']
#         result, confidence = self.TTS.calc_single_status(status_list, mode='strict')
#         self.assertEqual(result, 'YELLOW')
#         self.assertEqual(confidence, 1.0)
#         result, confidence = self.TTS.calc_single_status(status_list, mode='distr')
#         self.assertEqual(result, 'YELLOW')
#         self.assertEqual(confidence, 0.6666666666666666)
#         result, confidence = self.TTS.calc_single_status(status_list, mode='max')
#         self.assertEqual(result, 'YELLOW')
#         self.assertEqual(confidence, 1.0)
#         return


# if __name__ == '__main__':
#     unittest.main()
