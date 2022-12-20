  
""" Test Status Functionality """
import pytest
from mock import MagicMock
import src.systems.telemetry_test_suite as telemetry_test_suite
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

# run_tests tests
def test_run_tests_return_Status_object_based_upon_given_header_index_but_does_not_append_to_status_when_given_header_index_leads_to_empty_tests(mocker):
    # Arrange
    arg_header_index = MagicMock()
    arg_test_val = MagicMock()
    arg_sync_data = MagicMock()

    fake_bayesian = [MagicMock(),MagicMock()]

    expected_datafield = MagicMock()
    expected_result = telemetry_test_suite.Status.__new__(telemetry_test_suite.Status)

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)
    cut.tests = {arg_header_index:[]}
    cut.dataFields = {arg_header_index:expected_datafield}

    mocker.patch.object(cut, 'calc_single_status', return_value=fake_bayesian)
    mocker.patch('src.systems.telemetry_test_suite.Status', return_value = expected_result)

    # Act
    result = cut.run_tests(arg_header_index, arg_test_val, arg_sync_data)

    # Assert
    assert cut.calc_single_status.call_count == 1
    assert cut.calc_single_status.call_args_list[0].args == ([], )
    assert telemetry_test_suite.Status.call_count == 1
    assert telemetry_test_suite.Status.call_args_list[0].args == (expected_datafield, fake_bayesian[0], fake_bayesian[1])
    assert result == expected_result

def test_run_tests_return_Status_object_based_upon_given_header_index_appends_status_when_given_header_index_leads_to_a_single_test_not_named_SYNC(mocker):
    # Arrange
    arg_header_index = MagicMock()
    arg_test_val = MagicMock()
    arg_sync_data = MagicMock()

    fake_tests = [[str(MagicMock())]]
    for i in range(pytest.gen.randint(0,5)): # arbirary, from 0 to 5 test data points
        fake_tests[0].append(MagicMock())
    fake_stat = MagicMock()
    fake_mass_assigments = MagicMock()
    fake_bayesian = [MagicMock(),MagicMock()]

    expected_datafield = MagicMock()
    expected_result = telemetry_test_suite.Status.__new__(telemetry_test_suite.Status)

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)
    cut.tests = {arg_header_index:fake_tests}
    cut.dataFields = {arg_header_index:expected_datafield}
    cut.epsilon = MagicMock()
    
    # IMPORTANT: note, using sync function as an easy mock -- not really calling it here!!
    mocker.patch.object(cut, 'sync', return_value=(fake_stat, fake_mass_assigments))
    mocker.patch.object(cut, 'calc_single_status', return_value=fake_bayesian)
    mocker.patch('src.systems.telemetry_test_suite.Status', return_value = expected_result)

    cut.all_tests = {fake_tests[0][0]:cut.sync} # IMPORTANT: purposely set AFTER patch of cut's sync function
    
    # Act
    result = cut.run_tests(arg_header_index, arg_test_val, arg_sync_data)

    # Assert
    assert cut.sync.call_count == 1
    assert cut.sync.call_args_list[0].args == (arg_test_val, fake_tests[0][1:], cut.epsilon)
    assert cut.calc_single_status.call_count == 1
    assert cut.calc_single_status.call_args_list[0].args == ([fake_stat], )
    assert telemetry_test_suite.Status.call_count == 1
    assert telemetry_test_suite.Status.call_args_list[0].args == (expected_datafield, fake_bayesian[0], fake_bayesian[1])
    assert result == expected_result
  
def test_run_tests_return_Status_object_based_upon_given_header_index_appends_status_with_empty_data_var_not_in_sync_data_keys_when_given_header_index_leads_to_a_single_test_named_SYNC(mocker):
    # Arrange
    arg_header_index = MagicMock()
    arg_test_val = MagicMock()
    arg_sync_data = {}

    fake_tests = [['SYNC']]
    fake_var = MagicMock()
    fake_tests[0].append(fake_var)
    fake_stat = MagicMock()
    fake_mass_assigments = MagicMock()
    fake_bayesian = [MagicMock(),MagicMock()]

    expected_datafield = MagicMock()
    expected_result = telemetry_test_suite.Status.__new__(telemetry_test_suite.Status)

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)
    cut.tests = {arg_header_index:fake_tests}
    cut.dataFields = {arg_header_index:expected_datafield}
    cut.epsilon = MagicMock()
    
    mocker.patch.object(cut, 'sync', return_value=(fake_stat, fake_mass_assigments))
    mocker.patch.object(cut, 'calc_single_status', return_value=fake_bayesian)
    mocker.patch('src.systems.telemetry_test_suite.Status', return_value = expected_result)

    cut.all_tests = {fake_tests[0][0]:cut.sync} # IMPORTANT: purposely set AFTER patch of cut's sync function
    
    # Act
    result = cut.run_tests(arg_header_index, arg_test_val, arg_sync_data)

    # Assert
    assert cut.sync.call_count == 1
    assert cut.sync.call_args_list[0].args == (arg_test_val, [], cut.epsilon)
    assert cut.calc_single_status.call_count == 1
    assert cut.calc_single_status.call_args_list[0].args == ([fake_stat], )
    assert telemetry_test_suite.Status.call_count == 1
    assert telemetry_test_suite.Status.call_args_list[0].args == (expected_datafield, fake_bayesian[0], fake_bayesian[1])
    assert result == expected_result
    
def test_run_tests_return_Status_object_based_upon_given_header_index_appends_status_with_updated_data_var_in_sync_data_keys_when_given_header_index_leads_to_a_single_test_named_SYNC(mocker):
    # Arrange
    arg_header_index = MagicMock()
    arg_test_val = MagicMock()
    arg_sync_data = {}

    fake_tests = [['SYNC']]
    fake_var = MagicMock()
    fake_tests[0].append(fake_var)
    fake_updated_test_data = MagicMock()
    fake_stat = MagicMock()
    fake_mass_assigments = MagicMock()
    fake_bayesian = [MagicMock(),MagicMock()]

    arg_sync_data[fake_var] = fake_updated_test_data

    expected_datafield = MagicMock()
    expected_result = telemetry_test_suite.Status.__new__(telemetry_test_suite.Status)

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)
    cut.tests = {arg_header_index:fake_tests}
    cut.dataFields = {arg_header_index:expected_datafield}
    cut.epsilon = MagicMock()
    
    mocker.patch.object(cut, 'sync', return_value=(fake_stat, fake_mass_assigments))
    mocker.patch.object(cut, 'calc_single_status', return_value=fake_bayesian)
    mocker.patch('src.systems.telemetry_test_suite.Status', return_value = expected_result)

    cut.all_tests = {fake_tests[0][0]:cut.sync} # IMPORTANT: purposely set AFTER patch of cut's sync function
    
    # Act
    result = cut.run_tests(arg_header_index, arg_test_val, arg_sync_data)

    # Assert
    assert cut.sync.call_count == 1
    assert cut.sync.call_args_list[0].args == (arg_test_val, [fake_updated_test_data], cut.epsilon)
    assert cut.calc_single_status.call_count == 1
    assert cut.calc_single_status.call_args_list[0].args == ([fake_stat], )
    assert telemetry_test_suite.Status.call_count == 1
    assert telemetry_test_suite.Status.call_args_list[0].args == (expected_datafield, fake_bayesian[0], fake_bayesian[1])
    assert result == expected_result

def test_run_tests_return_Status_object_based_upon_given_header_index_appends_status_with_any_updates_where_vars_in_sync_data_keys_when_given_header_index_leads_to_multiple_tests(mocker):
    # Arrange
    arg_header_index = MagicMock()
    arg_test_val = MagicMock()
    arg_sync_data = {}

    num_fake_tests = pytest.gen.randint(1, 5) # arbitrary, from 1 to 5 tests (0 has own test)
    fake_tests = []
    fake_vars = []
    fake_sync_vars = []
    fake_stat = MagicMock()
    fake_mass_assigments = MagicMock()
    fake_bayesian = [MagicMock(),MagicMock()]
    
    expected_datafield = MagicMock()
    expected_result = telemetry_test_suite.Status.__new__(telemetry_test_suite.Status)
    expected_stats = []
    for i in range(num_fake_tests):
        expected_stats.append(fake_stat)

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)
    cut.tests = {arg_header_index:fake_tests}
    cut.dataFields = {arg_header_index:expected_datafield}
    cut.epsilon = MagicMock()
    
    mocker.patch.object(cut, 'sync', return_value=(fake_stat, fake_mass_assigments))
    mocker.patch.object(cut, 'calc_single_status', return_value=fake_bayesian)
    mocker.patch('src.systems.telemetry_test_suite.Status', return_value = expected_result)

    cut.all_tests = {'SYNC':cut.sync} # IMPORTANT: purposely set AFTER patch of cut's sync function
    
    # setup random input and results
    for i in range(num_fake_tests):
        fake_vars.append(MagicMock())
        rand_type = pytest.gen.randint(0, 2) # 3 choices, from 0 to 2
        if  rand_type == 0: # SYNC var exists
            fake_tests.append(['SYNC', fake_vars[i]])
            arg_sync_data[fake_vars[i]] = MagicMock()
            fake_sync_vars.append([arg_sync_data[fake_vars[i]]])
        elif rand_type == 1: # SYNC var not exists
            fake_tests.append(['SYNC', fake_vars[i]])
            fake_sync_vars.append([])
        else:
            fake_tests.append([str(MagicMock()), fake_vars[i]])
            fake_sync_vars.append([fake_vars[i]])
            cut.all_tests[fake_tests[i][0]] = cut.sync

    # Act
    result = cut.run_tests(arg_header_index, arg_test_val, arg_sync_data)

    # Assert
    assert cut.sync.call_count == num_fake_tests
    for i in range(num_fake_tests):
        assert cut.sync.call_args_list[i].args == (arg_test_val, fake_sync_vars[i], cut.epsilon)
    assert cut.calc_single_status.call_count == 1
    assert cut.calc_single_status.call_args_list[0].args == (expected_stats, )
    assert telemetry_test_suite.Status.call_count == 1
    assert telemetry_test_suite.Status.call_args_list[0].args == (expected_datafield, fake_bayesian[0], fake_bayesian[1])
    assert result == expected_result
  
# get_latest_result tests
def test_get_latest_results_returns_None_when_latest_results_is_None(mocker):
    # Arrange
    arg_field_name = MagicMock()

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)
    cut.latest_results = None

    # Act
    result = cut.get_latest_result(arg_field_name)

    # Assert
    assert result == None

def test_get_latest_results_returns_None_when_latest_results_is_filled(mocker):
    # Arrange
    arg_field_name = MagicMock()

    fake_hdr_index = MagicMock()

    expected_result = MagicMock()

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)
    cut.latest_results = {fake_hdr_index:expected_result}
    cut.dataFields = MagicMock()

    mocker.patch.object(cut.dataFields, 'index', return_value=fake_hdr_index)

    # Act
    result = cut.get_latest_result(arg_field_name)

    # Assert
    assert result == expected_result

# sync tests
def test_sync_returns_tuple_of_str_GREEN_and_list_containing_tuple_of_set_of_str_GREEN_and_1_pt_0():
    # Arrange
    arg_val = MagicMock()
    arg_test_params = MagicMock()
    arg_epsilon = MagicMock()

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    # Act
    result = cut.sync(arg_val, arg_test_params, arg_epsilon)

    # Assert
    assert result == ('GREEN', [({'GREEN'}, 1.0)])

# rotational tests
def test_rotational_returns_tuple_of_str_YELLOW_and_empty_list():
    # Arrange
    arg_val = MagicMock()
    arg_test_params = MagicMock()
    arg_epsilon = MagicMock()

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    # Act
    result = cut.rotational(arg_val, arg_test_params, arg_epsilon)

    # Assert
    assert result == ('YELLOW', [])

# state tests
def test_state_returns_tuple_of_str_GREEN_and_list_containing_tuple_of_set_of_str_GREEN_and_1_pt_0_when_int_val_is_in_range_test_params_0():
    # Arrange
    arg_test_params = []
    arg_epsilon = MagicMock()

    factor = 1
    if pytest.gen.randint(0,1) == 1:
        factor *= -1
     # arbitrary, from 0 to 200 with 50/50 change of negative 
    fake_mid_point = pytest.gen.randint(0, 200) * factor # arbitrary, from 0 to 200 with 50/50 change of negative 
    fake_green_tol = pytest.gen.randint(1, 50) # arbitrary, from 1 to 50 allowance in both directions from fake_mid_point

    arg_test_params.append(range((fake_mid_point - fake_green_tol), (fake_mid_point + fake_green_tol)))
    arg_test_params.append(MagicMock())
    arg_test_params.append(MagicMock())

    arg_val = pytest.gen.randint(0 - fake_green_tol, fake_green_tol - 1) + fake_mid_point # random val within green range
    if arg_val > 0:
        arg_val += pytest.gen.random() # make float by adding some random decimal
    else:
        arg_val -= pytest.gen.random() # make float by adding some random decimal

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    # Act
    result = cut.state(arg_val, arg_test_params, arg_epsilon)

    # Assert
    assert result == ('GREEN', [({'GREEN'}, 1.0)])

def test_state_returns_tuple_of_str_YELLOW_and_list_containing_tuple_of_set_of_str_YELLOW_and_1_pt_0_when_int_val_is_in_range_test_params_1_and_not_in_0():
    # Arrange
    arg_test_params = []
    arg_epsilon = MagicMock()

    factor = 1
    if pytest.gen.randint(0,1) == 1:
        factor *= -1
     # arbitrary, from 0 to 200 with 50/50 change of negative 
    fake_mid_point = pytest.gen.randint(0, 200) * factor # arbitrary, from 0 to 200 with 50/50 change of negative 
    fake_green_tol = pytest.gen.randint(1, 50) # arbitrary, from 1 to 50 allowance in both directions from fake_mid_point
    fake_yellow_tol = pytest.gen.randint(1, 20) + fake_green_tol # arbitrary, from 1 to 20 allowance in both directions from fake_mid_point + fake_green_tol

    arg_test_params.append(range((fake_mid_point - fake_green_tol), (fake_mid_point + fake_green_tol)))
    arg_test_params.append(range((fake_mid_point - fake_yellow_tol), (fake_mid_point + fake_yellow_tol)))
    arg_test_params.append(MagicMock())

    if pytest.gen.randint(0,1) == 1: 
        arg_val = pytest.gen.randint(fake_green_tol, fake_yellow_tol - 1) + fake_mid_point # random val within upper yellow range
    else:
        arg_val = pytest.gen.randint(0 - fake_yellow_tol, 0 - fake_green_tol - 1) + fake_mid_point # sometimes flip to lower yellow range   
    if arg_val > 0:
        arg_val += pytest.gen.random() # make float by adding some random decimal
    else:
        arg_val -= pytest.gen.random() # make float by adding some random decimal
    
    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    # Act
    result = cut.state(arg_val, arg_test_params, arg_epsilon)

    # Assert
    assert result == ('YELLOW', [({'YELLOW'}, 1.0)])
    
def test_state_returns_tuple_of_str_RED_and_list_containing_tuple_of_set_of_str_RED_and_1_pt_0_when_int_val_is_in_range_test_params_2_and_not_in_0_or_1():
    # Arrange
    arg_test_params = []
    arg_epsilon = MagicMock()

    factor = 1
    if pytest.gen.randint(0,1) == 1:
        factor *= -1
     # arbitrary, from 0 to 200 with 50/50 change of negative 
    fake_mid_point = pytest.gen.randint(0, 200) * factor # arbitrary, from 0 to 200 with 50/50 change of negative 
    fake_green_tol = pytest.gen.randint(1, 50) # arbitrary, from 1 to 50 allowance in both directions from fake_mid_point
    fake_yellow_tol = pytest.gen.randint(1, 20) + fake_green_tol # arbitrary, from 1 to 20 allowance in both directions from fake_mid_point + fake_green_tol
    fake_red_tol = pytest.gen.randint(1, 10) + fake_yellow_tol # arbitrary, from 1 to 10 allowance in both directions from fake_mid_point + fake_yellow_tol

    arg_test_params.append(range((fake_mid_point - fake_green_tol), (fake_mid_point + fake_green_tol)))
    arg_test_params.append(range((fake_mid_point - fake_yellow_tol), (fake_mid_point + fake_yellow_tol)))
    arg_test_params.append(range((fake_mid_point - fake_red_tol), (fake_mid_point + fake_red_tol)))

    if pytest.gen.randint(0,1) == 1: 
        arg_val = pytest.gen.randint(fake_yellow_tol, fake_red_tol - 1) + fake_mid_point # random val within upper red range
    else:
        arg_val = pytest.gen.randint(0 - fake_red_tol, 0 - fake_yellow_tol - 1) + fake_mid_point # sometimes flip to lower red range   
    if arg_val > 0:
        arg_val += pytest.gen.random() # make float by adding some random decimal
    else:
        arg_val -= pytest.gen.random() # make float by adding some random decimal
    
    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    # Act
    result = cut.state(arg_val, arg_test_params, arg_epsilon)

    # Assert
    assert result == ('RED', [({'RED'}, 1.0)])
     
def test_state_returns_tuple_of_str_3_dashes_and_list_containing_tuple_of_set_of_str_RED_YELLOW_and_GREEN_and_1_pt_0_when_int_val_is_in_not_in_any_range():
    # Arrange
    arg_test_params = []
    arg_epsilon = MagicMock()

    factor = 1
    if pytest.gen.randint(0,1) == 1:
        factor *= -1
     # arbitrary, from 0 to 200 with 50/50 change of negative 
    fake_mid_point = pytest.gen.randint(0, 200) * factor # arbitrary, from 0 to 200 with 50/50 change of negative 
    fake_green_tol = pytest.gen.randint(1, 50) # arbitrary, from 1 to 50 allowance in both directions from fake_mid_point
    fake_yellow_tol = pytest.gen.randint(1, 20) + fake_green_tol # arbitrary, from 1 to 20 allowance in both directions from fake_mid_point + fake_green_tol
    fake_red_tol = pytest.gen.randint(1, 10) + fake_yellow_tol # arbitrary, from 1 to 10 allowance in both directions from fake_mid_point + fake_yellow_tol

    arg_test_params.append(range((fake_mid_point - fake_green_tol), (fake_mid_point + fake_green_tol)))
    arg_test_params.append(range((fake_mid_point - fake_yellow_tol), (fake_mid_point + fake_yellow_tol)))
    arg_test_params.append(range((fake_mid_point - fake_red_tol), (fake_mid_point + fake_red_tol)))

    if pytest.gen.randint(0,1) == 1: 
        arg_val = fake_red_tol + fake_mid_point + 1 # random val outside upper red
    else:
        arg_val = 0 - fake_red_tol + fake_mid_point - 1 # random val outside lower red  
    if arg_val > 0:
        arg_val += pytest.gen.random() # make float by adding some random decimal
    else:
        arg_val -= pytest.gen.random() # make float by adding some random decimal
    
    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    # Act
    result = cut.state(arg_val, arg_test_params, arg_epsilon)

    # Assert
    assert result == ('---', [({'RED', 'YELLOW', 'GREEN'}, 1.0)])
    
# feasibility tests

# noop tests

# calc_single_status tests

# get_suite_status
def test_get_suite_status_raises_TypeError_when_latest_results_is_None():
    # Arrange
    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)
    cut.latest_results = None

    # Act
    with pytest.raises(TypeError) as e_info:
        result = cut.get_suite_status()

    # Assert
    assert e_info.match("'NoneType' object is not iterable")

def test_get_suite_status_returns_value_from_call_to_calc_single_status_when_it_is_given_empty_list_because_latest_results_are_empty(mocker):
    # Arrange
    expected_result = MagicMock()

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)
    cut.latest_results = []

    mocker.patch.object(cut, 'calc_single_status', return_value=expected_result)

    # Act
    result = cut.get_suite_status()

    # Assert
    assert cut.calc_single_status.call_count == 1
    assert cut.calc_single_status.call_args_list[0].args == ([], )
    assert result == expected_result

def test_get_suite_status_returns_value_from_call_to_calc_single_status_when_it_is_given_list_of_all_statuses_in_latest_results(mocker):
    # Arrange
    num_fake_results = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10 (0 has its own test)
    fake_latest_results = []
    fake_statuses = []

    for i in range(num_fake_results):
        fake_res = MagicMock()
        fake_status = MagicMock()

        mocker.patch.object(fake_res, 'get_status', return_value=fake_status)
        
        fake_latest_results.append(fake_res)
        fake_statuses.append(fake_status)

    expected_result = MagicMock()

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)
    cut.latest_results = fake_latest_results

    mocker.patch.object(cut, 'calc_single_status', return_value=expected_result)

    # Act
    result = cut.get_suite_status()

    # Assert
    assert cut.calc_single_status.call_count == 1
    assert cut.calc_single_status.call_args_list[0].args == (fake_statuses, )
    assert result == expected_result

# get_status_specific_mnemonics
# test_get_status_specific_mnemonics_raises_TypeError_when_latest_results_is_None was written because None is the init value for latest_results
def test_get_status_specific_mnemonics_raises_TypeError_when_latest_results_is_None(mocker):
     # Arrange
    arg_status = MagicMock()

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)
    cut.latest_results = None

    # Act
    with pytest.raises(TypeError) as e_info:
        result = cut.get_status_specific_mnemonics(arg_status)

    # Assert
    assert e_info.match("'NoneType' object is not iterable")

def test_get_status_specific_mnemonics_returns_empty_list_when_latest_results_is_empty(mocker):
    # Arrange
    arg_status = MagicMock()

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)
    cut.latest_results = []

    # Act
    result = cut.get_status_specific_mnemonics(arg_status)

    # Assert
    assert result == []

def test_get_status_specific_mnemonics_returns_the_only_name_in_latest_results_because_its_status_eq_given_status(mocker):
    # Arrange
    arg_status = MagicMock()

    fake_res = MagicMock()

    expected_name = str(MagicMock())

    mocker.patch.object(fake_res, 'get_status', return_value=arg_status)
    mocker.patch.object(fake_res, 'get_name', return_value=expected_name)

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)
    cut.latest_results = [fake_res]

    # Act
    result = cut.get_status_specific_mnemonics(arg_status)

    # Assert
    assert result == [expected_name]

def test_get_status_specific_mnemonics_returns_empty_list_latest_results_because_its_status_not_eq_given_status(mocker):
    # Arrange
    arg_status = MagicMock()

    fake_res = MagicMock()

    mocker.patch.object(fake_res, 'get_status', return_value=MagicMock())

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)
    cut.latest_results = [fake_res]

    # Act
    result = cut.get_status_specific_mnemonics(arg_status)

    # Assert
    assert result == []

def test_get_status_specific_mnemonics_returns_only_names_in_latest_results_where_status_matches_given_status(mocker):
    # Arrange
    arg_status = MagicMock()

    num_fake_results = pytest.gen.randint(2, 10) # arbitrary, from 2 to 10 (0 and 1 both have own test)
    num_fake_status_matches = pytest.gen.randint(1, num_fake_results - 1) # at least 1 match up to 1 less than all
    fake_latest_results = [False] * num_fake_results

    expected_names = []

    for i in pytest.gen.sample(range(len(fake_latest_results)), num_fake_status_matches):
        fake_latest_results[i] = True

    for i in range(len(fake_latest_results)):
        fake_res = MagicMock()
        if fake_latest_results[i] == True:
            fake_name = str(MagicMock())
            mocker.patch.object(fake_res, 'get_status', return_value=arg_status)
            mocker.patch.object(fake_res, 'get_name', return_value=fake_name)
            expected_names.append(fake_name)
        else:
            mocker.patch.object(fake_res, 'get_status', return_value=MagicMock())
        fake_latest_results[i] = fake_res
            
    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)
    cut.latest_results = fake_latest_results

    # Act
    result = cut.get_status_specific_mnemonics(arg_status)

    # Assert
    assert result == expected_names
    assert len(result) != len(fake_latest_results)

def test_get_status_specific_mnemonics_returns_all_names_in_latest_results_when_all_statuses_matches_given_status(mocker):
    # Arrange
    arg_status = MagicMock()

    num_fake_results = pytest.gen.randint(1, 10) # arbitrary, from 2 to 10 (0 and 1 both have own test)
    fake_latest_results = []

    expected_names = []

    for i in range(num_fake_results):
        fake_res = MagicMock()
        fake_name = str(MagicMock())
        mocker.patch.object(fake_res, 'get_status', return_value=arg_status)
        mocker.patch.object(fake_res, 'get_name', return_value=fake_name)
        fake_latest_results.append(fake_res)
        expected_names.append(fake_name)
            
    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)
    cut.latest_results = fake_latest_results

    # Act
    result = cut.get_status_specific_mnemonics(arg_status)

    # Assert
    assert result == expected_names
    assert len(result) == len(fake_latest_results)

def test_get_status_specific_mnemonics_default_given_status_is_str_RED(mocker):
    # Arrange
    fake_res = MagicMock()

    expected_name = str(MagicMock())

    mocker.patch.object(fake_res, 'get_status', return_value='RED')
    mocker.patch.object(fake_res, 'get_name', return_value=expected_name)

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)
    cut.latest_results = [fake_res]

    # Act
    result = cut.get_status_specific_mnemonics()

    # Assert
    assert result == [expected_name]

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
