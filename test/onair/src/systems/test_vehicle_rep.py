# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

""" Test VehicleRepresentation Functionality """
import pytest
from mock import MagicMock

import onair.src.systems.vehicle_rep as vehicle_rep
from onair.src.systems.vehicle_rep import VehicleRepresentation

# __init__ tests
def test_VehicleRepresentation__init__asserts_when_len_given_headers_is_not_eq_to_len_given_tests(mocker):
    # Arrange
    arg_headers = MagicMock()
    arg_tests = MagicMock()

    fake_len = []
    fake_len.append(pytest.gen.randint(0, 100)) # arbitrary, from 0 to 100 size
    fake_len.append(fake_len[0])
    while fake_len[1] == fake_len[0]: # need a value not equal for test to pass
        fake_len[1] = pytest.gen.randint(0, 100) # arbitrary, same as fake_len_headers

    cut = VehicleRepresentation.__new__(VehicleRepresentation)

    mocker.patch(vehicle_rep.__name__ + '.len', side_effect=fake_len)
    # Act
    with pytest.raises(AssertionError) as e_info:
        cut.__init__(arg_headers, arg_tests)
    
    # Assert
    assert vehicle_rep.len.call_count == 2
    call_list = set({})
    [call_list.add(vehicle_rep.len.call_args_list[i].args) for i in range(len(vehicle_rep.len.call_args_list))]
    assert call_list == {(arg_headers, ), (arg_tests, )}
    assert e_info.match('')

def test_VehicleRepresentation__init__sets_status_to_Status_with_str_MISSION_and_headers_to_given_headers_and_test_suite_to_TelemetryTestSuite_with_given_headers_and_tests_and_curr_data_to_all_empty_step_len_of_headers(mocker):
    # Arrange
    arg_headers = MagicMock()
    arg_tests = MagicMock()

    fake_len = pytest.gen.randint(0, 100) # arbitrary, 0 to 100 items
    fake_status = MagicMock()
    fake_test_suite = MagicMock()

    cut = VehicleRepresentation.__new__(VehicleRepresentation)

    mocker.patch(vehicle_rep.__name__ + '.len', return_value=fake_len)
    mocker.patch(vehicle_rep.__name__ + '.Status', return_value=fake_status)
    mocker.patch(vehicle_rep.__name__ + '.TelemetryTestSuite', return_value=fake_test_suite)
    
    # Act
    cut.__init__(arg_headers, arg_tests)

    # Assert
    assert vehicle_rep.Status.call_count == 1
    assert vehicle_rep.Status.call_args_list[0].args == ('MISSION', )
    assert cut.status == fake_status
    assert cut.headers == arg_headers
    assert vehicle_rep.TelemetryTestSuite.call_count == 1
    assert vehicle_rep.TelemetryTestSuite.call_args_list[0].args == (arg_headers, arg_tests)
    assert cut.test_suite == fake_test_suite
    assert cut.curr_data == ['-'] * fake_len

# NOTE: commonly each optional arg is tested, but because their sizes must be equal testing both at once
def test_VehicleRepresentation__init__default_given_headers_and_tests_are_both_empty_list(mocker):
    # Arrange
    cut = VehicleRepresentation.__new__(VehicleRepresentation)

    mocker.patch(vehicle_rep.__name__ + '.Status')
    mocker.patch(vehicle_rep.__name__ + '.TelemetryTestSuite')
    
    # Act
    cut.__init__()

    # Assert
    assert cut.headers == []
    assert vehicle_rep.TelemetryTestSuite.call_count == 1
    assert vehicle_rep.TelemetryTestSuite.call_args_list[0].args == ([], [])
    assert cut.curr_data == ['-'] * 0

# update tests
def test_VehicleRepresentation_update_does_not_set_any_curr_data_when_given_frame_is_vacant_and_executes_suite_with_given_frame_and_sets_status_with_suite_status(mocker):
    # Arrange
    arg_frame = []

    cut = VehicleRepresentation.__new__(VehicleRepresentation)
    cut.test_suite = MagicMock()
    cut.status = MagicMock()

    fake_suite_status = []
    for i in range(pytest.gen.randint(1, 10)): # arbitrary, from 1 to 10 status items
        fake_suite_status.append(MagicMock())

    mocker.patch.object(cut.test_suite, 'execute_suite')
    mocker.patch.object(cut.test_suite, 'get_suite_status', return_value=fake_suite_status)
    mocker.patch.object(cut.status, 'set_status')
    
    # Act
    cut.update(arg_frame)

    # Assert
    assert cut.test_suite.execute_suite.call_count == 1
    assert cut.test_suite.execute_suite.call_args_list[0].args == (arg_frame, )
    assert cut.test_suite.get_suite_status.call_count == 1
    assert cut.test_suite.get_suite_status.call_args_list[0].args == ()
    assert cut.status.set_status.call_count == 1
    assert cut.status.set_status.call_args_list[0].args == tuple(fake_suite_status)
    
def test_VehicleRepresentation_update_does_not_set_any_curr_data_when_given_frame_is_all_empty_step_and_executes_suite_with_given_frame_and_sets_status_with_suite_status(mocker):
    # Arrange
    arg_frame = []

    num_fake_empty_steps = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10 (0 has own test)
    for i in range(num_fake_empty_steps):
        arg_frame.append('-')

    cut = VehicleRepresentation.__new__(VehicleRepresentation)
    cut.test_suite = MagicMock()
    cut.status = MagicMock()
    cut.curr_data = []

    fake_suite_status = []
    for i in range(pytest.gen.randint(1, 10)): # arbitrary, from 1 to 10 status items
        fake_suite_status.append(MagicMock())

    mocker.patch.object(cut.test_suite, 'execute_suite')
    mocker.patch.object(cut.test_suite, 'get_suite_status', return_value=fake_suite_status)
    mocker.patch.object(cut.status, 'set_status')
    
    # Act
    cut.update(arg_frame)

    # Assert
    assert len(arg_frame) == num_fake_empty_steps
    assert cut.curr_data == []
    assert cut.test_suite.execute_suite.call_count == 1
    assert cut.test_suite.execute_suite.call_args_list[0].args == (arg_frame, )
    assert cut.test_suite.get_suite_status.call_count == 1
    assert cut.test_suite.get_suite_status.call_args_list[0].args == ()
    assert cut.status.set_status.call_count == 1
    assert cut.status.set_status.call_args_list[0].args == tuple(fake_suite_status)
    
def test_VehicleRepresentation_update_does_puts_all_frame_data_into_curr_data_when_none_are_empty_step_and_executes_suite_with_given_frame_and_sets_status_with_suite_status(mocker):
    # Arrange
    arg_frame = []

    num_fake_full_steps = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10 (0 has own test)
    for i in range(num_fake_full_steps):
        arg_frame.append(MagicMock())

    cut = VehicleRepresentation.__new__(VehicleRepresentation)
    cut.test_suite = MagicMock()
    cut.status = MagicMock()
    cut.curr_data = [MagicMock()] * num_fake_full_steps

    fake_suite_status = []
    for i in range(pytest.gen.randint(1, 10)): # arbitrary, from 1 to 10 status items
        fake_suite_status.append(MagicMock())

    mocker.patch.object(cut.test_suite, 'execute_suite')
    mocker.patch.object(cut.test_suite, 'get_suite_status', return_value=fake_suite_status)
    mocker.patch.object(cut.status, 'set_status')
    
    # Act
    cut.update(arg_frame)

    # Assert
    assert len(arg_frame) == num_fake_full_steps
    assert cut.curr_data == arg_frame
    assert cut.test_suite.execute_suite.call_count == 1
    assert cut.test_suite.execute_suite.call_args_list[0].args == (arg_frame, )
    assert cut.test_suite.get_suite_status.call_count == 1
    assert cut.test_suite.get_suite_status.call_args_list[0].args == ()
    assert cut.status.set_status.call_count == 1
    assert cut.status.set_status.call_args_list[0].args == tuple(fake_suite_status)
    
def test_VehicleRepresentation_update_puts_frame_data_into_curr_data_at_same_list_location_unless_data_is_empty_step_then_leaves_curr_data_that_location_alone_and_executes_suite_with_given_frame_and_sets_status_with_suite_status(mocker):
    # Arrange
    arg_frame = []

    num_fake_total_steps = pytest.gen.randint(2, 10) # arbitrary, from 2 to 10 (must have 2 to have at least one of each type)
    for i in range(num_fake_total_steps):
        arg_frame.append(MagicMock())
    location_fake_empty_steps = pytest.gen.sample(list(range(num_fake_total_steps)), pytest.gen.randint(1, num_fake_total_steps - 1)) # sample from a list of all numbers up to total then take from 1 to up to 1 less than total
    for i in location_fake_empty_steps:
        arg_frame[i] = '-'

    cut = VehicleRepresentation.__new__(VehicleRepresentation)
    cut.test_suite = MagicMock()
    cut.status = MagicMock()
    unchanged_data = MagicMock()
    cut.curr_data = [unchanged_data] * num_fake_total_steps

    fake_suite_status = []
    for i in range(pytest.gen.randint(1, 10)): # arbitrary, from 1 to 10 status items
        fake_suite_status.append(MagicMock())

    mocker.patch.object(cut.test_suite, 'execute_suite')
    mocker.patch.object(cut.test_suite, 'get_suite_status', return_value=fake_suite_status)
    mocker.patch.object(cut.status, 'set_status')
    
    # Act
    cut.update(arg_frame)

    # Assert
    assert len(arg_frame) == num_fake_total_steps
    for i in range(len(cut.curr_data)):
        if location_fake_empty_steps.count(i):
            assert cut.curr_data[i] == unchanged_data
        else:
            assert cut.curr_data[i] == arg_frame[i]
    assert cut.test_suite.execute_suite.call_count == 1
    assert cut.test_suite.execute_suite.call_args_list[0].args == (arg_frame, )
    assert cut.test_suite.get_suite_status.call_count == 1
    assert cut.test_suite.get_suite_status.call_args_list[0].args == ()
    assert cut.status.set_status.call_count == 1
    assert cut.status.set_status.call_args_list[0].args == tuple(fake_suite_status)
    
def test_VehicleRepresentation_update_puts_frame_data_into_curr_data_at_same_list_location_unless_data_is_empty_step_then_leaves_curr_data_that_location_alone_including_locations_in_curr_data_that_do_not_exist_in_frame_and_executes_suite_with_given_frame_and_sets_status_with_suite_status(mocker):
    # Arrange
    arg_frame = []

    num_fake_total_steps = pytest.gen.randint(2, 10) # arbitrary, from 2 to 10 (must have 2 to have at least one of each type)
    for i in range(num_fake_total_steps):
        arg_frame.append(MagicMock())
    location_fake_empty_steps = pytest.gen.sample(list(range(num_fake_total_steps)), pytest.gen.randint(1, num_fake_total_steps - 1)) # sample from a list of all numbers up to total then take from 1 to up to 1 less than total
    for i in location_fake_empty_steps:
        arg_frame[i] = '-'

    cut = VehicleRepresentation.__new__(VehicleRepresentation)
    cut.test_suite = MagicMock()
    cut.status = MagicMock()
    unchanged_data = MagicMock()
    cut.curr_data = [unchanged_data] * (num_fake_total_steps + pytest.gen.randint(1, 10)) # arbitrary, from 1 to 10 extra items over frame

    fake_suite_status = []
    for i in range(pytest.gen.randint(1, 10)): # arbitrary, from 1 to 10 status items
        fake_suite_status.append(MagicMock())

    mocker.patch.object(cut.test_suite, 'execute_suite')
    mocker.patch.object(cut.test_suite, 'get_suite_status', return_value=fake_suite_status)
    mocker.patch.object(cut.status, 'set_status')
    
    # Act
    cut.update(arg_frame)

    # Assert
    assert len(arg_frame) == num_fake_total_steps
    assert len(cut.curr_data) > len(arg_frame)
    for i in range(len(cut.curr_data)):
        if i >= len(arg_frame) or location_fake_empty_steps.count(i):
            assert cut.curr_data[i] == unchanged_data
        else:
            assert cut.curr_data[i] == arg_frame[i]
    assert cut.test_suite.execute_suite.call_count == 1
    assert cut.test_suite.execute_suite.call_args_list[0].args == (arg_frame, )
    assert cut.test_suite.get_suite_status.call_count == 1
    assert cut.test_suite.get_suite_status.call_args_list[0].args == ()
    assert cut.status.set_status.call_count == 1
    assert cut.status.set_status.call_args_list[0].args == tuple(fake_suite_status)
    
def test_VehicleRepresentation_update_raises_IndexError_when_frame_location_size_with_relevant_data_extends_beyond_curr_data_size(mocker):
    # Arrange
    arg_frame = []

    num_fake_total_steps = pytest.gen.randint(2, 10) # arbitrary, from 2 to 10 (must have 2 to have at least one of each type)
    for i in range(num_fake_total_steps):
        arg_frame.append(MagicMock())

    cut = VehicleRepresentation.__new__(VehicleRepresentation)
    cut.test_suite = MagicMock()
    cut.status = MagicMock()
    unchanged_data = MagicMock()
    cut.curr_data = [unchanged_data] * (num_fake_total_steps - 1) # - 1 ensures less than with min of 1

    fake_suite_status = []
    for i in range(pytest.gen.randint(1, 10)): # arbitrary, from 1 to 10 status items
        fake_suite_status.append(MagicMock())

    mocker.patch.object(cut.test_suite, 'execute_suite')
    mocker.patch.object(cut.test_suite, 'get_suite_status', return_value=fake_suite_status)
    mocker.patch.object(cut.status, 'set_status')
    
    # Act
    with pytest.raises(IndexError) as e_info:
        cut.update(arg_frame)

    # Assert
    assert e_info.match('list assignment index out of range')
    
def test_VehicleRepresentation_update_does_not_raise_IndexError_when_frame_location_size_with_only_empty_steps_extends_beyond_curr_data_size(mocker):
    # Arrange
    arg_frame = []

    num_fake_total_steps = pytest.gen.randint(2, 10) # arbitrary, from 2 to 10 (must have 2 to have at least one of each type)
    for i in range(num_fake_total_steps):
        arg_frame.append(MagicMock())

    cut = VehicleRepresentation.__new__(VehicleRepresentation)
    cut.test_suite = MagicMock()
    cut.status = MagicMock()
    unchanged_data = MagicMock()
    fake_curr_data_size = pytest.gen.randint(1, num_fake_total_steps - 1) # from 1 to 1 less than total
    cut.curr_data = [unchanged_data] * fake_curr_data_size

    for i in range(fake_curr_data_size, len(arg_frame)):
        arg_frame[i] = '-'

    fake_suite_status = []
    for i in range(pytest.gen.randint(1, 10)): # arbitrary, from 1 to 10 status items
        fake_suite_status.append(MagicMock())
    fake_non_IndexError_message = str(MagicMock())

    mocker.patch.object(cut.test_suite, 'execute_suite', side_effect=Exception(fake_non_IndexError_message)) # testing short circuit that is provable to not be the IndexError

    # Act
    with pytest.raises(Exception) as e_info:
        cut.update(arg_frame)

    # Assert
    assert e_info.match(fake_non_IndexError_message)

# get_headers
def test_VehicleRepresentation_get_headers_returns_headers():
    # Arrange
    expected_result = MagicMock()

    cut = VehicleRepresentation.__new__(VehicleRepresentation)
    cut.headers = expected_result

    # Act
    result = cut.get_headers()

    # Assert
    assert result == expected_result

# get_current_faulting_mnemonics tests, return_value=fake_suite_status
def test_VehicleRepresentation_get_current_faulting_mnemonics_returns_test_suite_call_get_status_specific_mnemonics(mocker):
    # Arrange
    expected_result = MagicMock()

    cut = VehicleRepresentation.__new__(VehicleRepresentation)
    cut.test_suite = MagicMock()

    mocker.patch.object(cut.test_suite, 'get_status_specific_mnemonics', return_value=expected_result)

    # Act
    result = cut.get_current_faulting_mnemonics()

    # Assert
    assert cut.test_suite.get_status_specific_mnemonics.call_count == 1
    assert cut.test_suite.get_status_specific_mnemonics.call_args_list[0].args == ()
    assert result == expected_result

# get_current_data tests
def test_VehicleRepresentation_get_current_data_returns_curr_data():
    # Arrange
    expected_result = MagicMock()

    cut = VehicleRepresentation.__new__(VehicleRepresentation)
    cut.curr_data = expected_result

    # Act
    result = cut.get_current_data()

    # Assert
    assert result == expected_result

# get_current_time tests
def test_VehicleRepresentation_get_current_time_returns_curr_data_item_0():
    # Arrange
    expected_result = MagicMock()

    cut = VehicleRepresentation.__new__(VehicleRepresentation)
    cut.curr_data = []
    cut.curr_data.append(expected_result)

    # Act
    result = cut.get_current_time()

    # Assert
    assert result == expected_result

# get_status tests
def test_VehicleRepresentation_get_status_returns_status_call_to_get_status(mocker):
    # Arrange
    expected_result = MagicMock()

    cut = VehicleRepresentation.__new__(VehicleRepresentation)
    cut.status = MagicMock()

    mocker.patch.object(cut.status, 'get_status', return_value=expected_result)

    # Act
    result = cut.get_status()

    # Assert
    assert cut.status.get_status.call_count == 1
    assert cut.status.get_status.call_args_list[0].args == ()
    assert result == expected_result

# get_bayesian_status tests
def test_VehicleRepresentation_get_bayesian_status_returns_status_call_to_get_bayesian_status(mocker):
    # Arrange
    expected_result = MagicMock()

    cut = VehicleRepresentation.__new__(VehicleRepresentation)
    cut.status = MagicMock()

    mocker.patch.object(cut.status, 'get_bayesian_status', return_value=expected_result)

    # Act
    result = cut.get_bayesian_status()

    # Assert
    assert cut.status.get_bayesian_status.call_count == 1
    assert cut.status.get_bayesian_status.call_args_list[0].args == ()
    assert result == expected_result

# get_batch_status_reports tests
def test_VehicleRepresentation_get_batch_status_reports_returngets_None():
    # Arrange
    arg_batch_data = MagicMock()

    expected_result = None

    cut = VehicleRepresentation.__new__(VehicleRepresentation)

    # Act
    result = cut.get_batch_status_reports(arg_batch_data)

    # Assert
    assert result == expected_result