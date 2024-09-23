# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

""" Test Status Functionality """
import pytest
from unittest.mock import MagicMock

import onair.src.systems.telemetry_test_suite as telemetry_test_suite
from onair.src.systems.telemetry_test_suite import TelemetryTestSuite


# __init__ tests
def test_TelemetryTestSuite__init__sets_the_expected_values_with_given_headers_and_tests(
    mocker,
):
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
    assert (
        cut.epsilon == 1 / 100000
    )  # production codes notes this value as needing intelligent definition
    assert cut.all_tests == {
        "STATE": cut.state,
        "FEASIBILITY": cut.feasibility,
        "NOOP": cut.noop,
    }


def test_TelemetryTestSuite__init__default_arg_tests_is_empty_list(mocker):
    # Arrange
    arg_headers = MagicMock()

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    # Act
    cut.__init__(arg_headers)

    # Assert
    assert cut.tests == []


def test_TelemetryTestSuite__init__default_arg_headers_is_empty_list(mocker):
    # Arrange
    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    # Act
    cut.__init__()

    # Assert
    assert cut.dataFields == []


# execute_suite tests
def test_TelemetryTestSuite_execute_suite_sets_the_latest_results_to_empty_list_when_updated_frame_len_is_0(
    mocker,
):
    # Arrange
    arg_update_frame = ""  # empty string for len of 0
    arg_sync_data = MagicMock()

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    # Act
    cut.execute_suite(arg_update_frame, arg_sync_data)

    # Assert
    assert cut.latest_results == []


def test_TelemetryTestSuite_execute_suite_sets_latests_results_to_list_of_run_tests_for_each_item_in_given_updated_frame(
    mocker,
):
    # Arrange
    arg_update_frame = []
    arg_sync_data = MagicMock()

    num_items_in_update = pytest.gen.randint(1, 10)  # arbitrary, from 1 to 10
    expected_results = []

    for i in range(num_items_in_update):
        arg_update_frame.append(MagicMock())
        expected_results.append(MagicMock())

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    mocker.patch.object(cut, "run_tests", side_effect=expected_results)

    # Act
    cut.execute_suite(arg_update_frame, arg_sync_data)

    # Assert
    assert cut.run_tests.call_count == num_items_in_update
    for i in range(num_items_in_update):
        assert cut.run_tests.call_args_list[i].args == (
            i,
            arg_update_frame[i],
            arg_sync_data,
        )
    assert cut.latest_results == expected_results


def test_TelemetryTestSuite_execute_suite_default_arg_sync_data_is_empty_map(mocker):
    # Arrange
    arg_update_frame = [MagicMock()]

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    mocker.patch.object(cut, "run_tests", return_value=86)  # arbitrary 86

    # Act
    cut.execute_suite(arg_update_frame)

    # Assert
    assert cut.run_tests.call_args_list[0].args == (0, arg_update_frame[0], {})


# run_tests tests
def test_TelemetryTestSuite_run_tests_return_Status_object_based_upon_given_header_index_but_does_not_append_to_status_when_given_header_index_leads_to_empty_tests(
    mocker,
):
    # Arrange
    arg_header_index = MagicMock()
    arg_test_val = MagicMock()
    arg_sync_data = MagicMock()

    fake_bayesian = [MagicMock(), MagicMock()]

    expected_datafield = MagicMock()
    expected_result = telemetry_test_suite.Status.__new__(telemetry_test_suite.Status)

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)
    cut.tests = {arg_header_index: []}
    cut.dataFields = {arg_header_index: expected_datafield}

    mocker.patch.object(cut, "calc_single_status", return_value=fake_bayesian)
    mocker.patch(
        telemetry_test_suite.__name__ + ".Status", return_value=expected_result
    )

    # Act
    result = cut.run_tests(arg_header_index, arg_test_val, arg_sync_data)

    # Assert
    assert cut.calc_single_status.call_count == 1
    assert cut.calc_single_status.call_args_list[0].args == ([],)
    assert telemetry_test_suite.Status.call_count == 1
    assert telemetry_test_suite.Status.call_args_list[0].args == (
        expected_datafield,
        fake_bayesian[0],
        fake_bayesian[1],
    )
    assert result == expected_result


def test_TelemetryTestSuite_run_tests_return_Status_object_based_upon_given_header_index_appends_status_when_given_header_index_leads_to_a_single_test_not_named_SYNC(
    mocker,
):
    # Arrange
    arg_header_index = MagicMock()
    arg_test_val = MagicMock()
    arg_sync_data = MagicMock()

    fake_tests = [[str(MagicMock())]]
    for i in range(pytest.gen.randint(0, 5)):  # arbirary, from 0 to 5 test data points
        fake_tests[0].append(MagicMock())
    fake_stat = MagicMock()
    fake_mass_assigments = MagicMock()
    fake_bayesian = [MagicMock(), MagicMock()]

    expected_datafield = MagicMock()
    expected_result = telemetry_test_suite.Status.__new__(telemetry_test_suite.Status)

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)
    cut.tests = {arg_header_index: fake_tests}
    cut.dataFields = {arg_header_index: expected_datafield}
    cut.epsilon = MagicMock()

    # IMPORTANT: note, using state function as an easy mock -- not really calling it here!!
    mocker.patch.object(cut, "state", return_value=(fake_stat, fake_mass_assigments))
    mocker.patch.object(cut, "calc_single_status", return_value=fake_bayesian)
    mocker.patch(
        telemetry_test_suite.__name__ + ".Status", return_value=expected_result
    )

    cut.all_tests = {
        fake_tests[0][0]: cut.state
    }  # IMPORTANT: purposely set AFTER patch of cut's sync function

    # Act
    result = cut.run_tests(arg_header_index, arg_test_val, arg_sync_data)

    # Assert
    assert cut.state.call_count == 1
    assert cut.state.call_args_list[0].args == (
        arg_test_val,
        fake_tests[0][1:],
        cut.epsilon,
    )
    assert cut.calc_single_status.call_count == 1
    assert cut.calc_single_status.call_args_list[0].args == ([fake_stat],)
    assert telemetry_test_suite.Status.call_count == 1
    assert telemetry_test_suite.Status.call_args_list[0].args == (
        expected_datafield,
        fake_bayesian[0],
        fake_bayesian[1],
    )
    assert result == expected_result


def test_TelemetryTestSuite_run_tests_return_Status_object_based_upon_given_header_index_appends_status_with_any_updates_where_vars_in_sync_data_keys_when_given_header_index_leads_to_multiple_tests(
    mocker,
):
    # Arrange
    arg_header_index = MagicMock()
    arg_test_val = MagicMock()
    arg_sync_data = {}

    num_fake_tests = pytest.gen.randint(
        1, 5
    )  # arbitrary, from 1 to 5 tests (0 has own test)
    fake_tests = []
    fake_vars = []
    fake_sync_vars = []
    fake_stat = MagicMock()
    fake_mass_assigments = MagicMock()
    fake_bayesian = [MagicMock(), MagicMock()]

    expected_datafield = MagicMock()
    expected_result = telemetry_test_suite.Status.__new__(telemetry_test_suite.Status)
    expected_stats = []
    for i in range(num_fake_tests):
        expected_stats.append(fake_stat)

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)
    cut.tests = {arg_header_index: fake_tests}
    cut.dataFields = {arg_header_index: expected_datafield}
    cut.epsilon = MagicMock()

    mocker.patch.object(cut, "state", return_value=(fake_stat, fake_mass_assigments))
    mocker.patch.object(cut, "calc_single_status", return_value=fake_bayesian)
    mocker.patch(
        telemetry_test_suite.__name__ + ".Status", return_value=expected_result
    )

    cut.all_tests = {
        "STATE": cut.state
    }  # IMPORTANT: purposely set AFTER patch of cut's state function

    # setup random input and results
    for i in range(num_fake_tests):
        fake_vars.append(MagicMock())
        fake_tests.append([str(MagicMock()), fake_vars[i]])
        fake_sync_vars.append([fake_vars[i]])
        cut.all_tests[fake_tests[i][0]] = cut.state

    # Act
    result = cut.run_tests(arg_header_index, arg_test_val, arg_sync_data)

    # Assert
    assert cut.state.call_count == num_fake_tests
    for i in range(num_fake_tests):
        assert cut.state.call_args_list[i].args == (
            arg_test_val,
            fake_sync_vars[i],
            cut.epsilon,
        )
    assert cut.calc_single_status.call_count == 1
    assert cut.calc_single_status.call_args_list[0].args == (expected_stats,)
    assert telemetry_test_suite.Status.call_count == 1
    assert telemetry_test_suite.Status.call_args_list[0].args == (
        expected_datafield,
        fake_bayesian[0],
        fake_bayesian[1],
    )
    assert result == expected_result


# get_latest_result tests
def test_TelemetryTestSuite_get_latest_results_returns_None_when_latest_results_is_None():
    # Arrange
    arg_field_name = MagicMock()

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)
    cut.latest_results = None

    # Act
    result = cut.get_latest_result(arg_field_name)

    # Assert
    assert result == None


def test_TelemetryTestSuite_get_latest_results_returns_None_when_latest_results_is_filled(
    mocker,
):
    # Arrange
    arg_field_name = MagicMock()

    fake_hdr_index = MagicMock()

    expected_result = MagicMock()

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)
    cut.latest_results = {fake_hdr_index: expected_result}
    cut.dataFields = MagicMock()

    mocker.patch.object(cut.dataFields, "index", return_value=fake_hdr_index)

    # Act
    result = cut.get_latest_result(arg_field_name)

    # Assert
    assert result == expected_result


# state tests
def test_TelemetryTestSuite_state_returns_tuple_of_str_GREEN_and_list_containing_tuple_of_set_of_str_GREEN_and_1_pt_0_when_int_val_is_in_range_test_params_0():
    # Arrange
    arg_test_params = []
    arg_epsilon = MagicMock()

    factor = 1
    if pytest.gen.randint(0, 1) == 1:
        factor *= -1
    # arbitrary, from 0 to 200 with 50/50 change of negative
    fake_mid_point = (
        pytest.gen.randint(0, 200) * factor
    )  # arbitrary, from 0 to 200 with 50/50 change of negative
    fake_green_tol = pytest.gen.randint(
        1, 50
    )  # arbitrary, from 1 to 50 allowance in both directions from fake_mid_point

    arg_test_params.append(
        range((fake_mid_point - fake_green_tol), (fake_mid_point + fake_green_tol))
    )
    arg_test_params.append(MagicMock())
    arg_test_params.append(MagicMock())

    arg_val = (
        pytest.gen.randint(0 - fake_green_tol, fake_green_tol - 1) + fake_mid_point
    )  # random val within green range
    if arg_val > 0:
        arg_val += pytest.gen.random()  # make float by adding some random decimal
    else:
        arg_val -= pytest.gen.random()  # make float by adding some random decimal

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    # Act
    result = cut.state(arg_val, arg_test_params, arg_epsilon)

    # Assert
    assert result == ("GREEN", [({"GREEN"}, 1.0)])


def test_TelemetryTestSuite_state_returns_tuple_of_str_YELLOW_and_list_containing_tuple_of_set_of_str_YELLOW_and_1_pt_0_when_int_val_is_in_range_test_params_1_and_not_in_0():
    # Arrange
    arg_test_params = []
    arg_epsilon = MagicMock()

    factor = 1
    if pytest.gen.randint(0, 1) == 1:
        factor *= -1
    # arbitrary, from 0 to 200 with 50/50 change of negative
    fake_mid_point = (
        pytest.gen.randint(0, 200) * factor
    )  # arbitrary, from 0 to 200 with 50/50 change of negative
    fake_green_tol = pytest.gen.randint(
        1, 50
    )  # arbitrary, from 1 to 50 allowance in both directions from fake_mid_point
    fake_yellow_tol = (
        pytest.gen.randint(1, 20) + fake_green_tol
    )  # arbitrary, from 1 to 20 allowance in both directions from fake_mid_point + fake_green_tol

    arg_test_params.append(
        range((fake_mid_point - fake_green_tol), (fake_mid_point + fake_green_tol))
    )
    arg_test_params.append(
        range((fake_mid_point - fake_yellow_tol), (fake_mid_point + fake_yellow_tol))
    )
    arg_test_params.append(MagicMock())

    if pytest.gen.randint(0, 1) == 1:
        arg_val = (
            pytest.gen.randint(fake_green_tol, fake_yellow_tol - 1) + fake_mid_point
        )  # random val within upper yellow range
    else:
        arg_val = (
            pytest.gen.randint(0 - fake_yellow_tol, 0 - fake_green_tol - 1)
            + fake_mid_point
        )  # sometimes flip to lower yellow range
    if arg_val > 0:
        arg_val += pytest.gen.random()  # make float by adding some random decimal
    else:
        arg_val -= pytest.gen.random()  # make float by adding some random decimal

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    # Act
    result = cut.state(arg_val, arg_test_params, arg_epsilon)

    # Assert
    assert result == ("YELLOW", [({"YELLOW"}, 1.0)])


def test_TelemetryTestSuite_state_returns_tuple_of_str_RED_and_list_containing_tuple_of_set_of_str_RED_and_1_pt_0_when_int_val_is_in_range_test_params_2_and_not_in_0_or_1():
    # Arrange
    arg_test_params = []
    arg_epsilon = MagicMock()

    factor = 1
    if pytest.gen.randint(0, 1) == 1:
        factor *= -1
    # arbitrary, from 0 to 200 with 50/50 change of negative
    fake_mid_point = (
        pytest.gen.randint(0, 200) * factor
    )  # arbitrary, from 0 to 200 with 50/50 change of negative
    fake_green_tol = pytest.gen.randint(
        1, 50
    )  # arbitrary, from 1 to 50 allowance in both directions from fake_mid_point
    fake_yellow_tol = (
        pytest.gen.randint(1, 20) + fake_green_tol
    )  # arbitrary, from 1 to 20 allowance in both directions from fake_mid_point + fake_green_tol
    fake_red_tol = (
        pytest.gen.randint(1, 10) + fake_yellow_tol
    )  # arbitrary, from 1 to 10 allowance in both directions from fake_mid_point + fake_yellow_tol

    arg_test_params.append(
        range((fake_mid_point - fake_green_tol), (fake_mid_point + fake_green_tol))
    )
    arg_test_params.append(
        range((fake_mid_point - fake_yellow_tol), (fake_mid_point + fake_yellow_tol))
    )
    arg_test_params.append(
        range((fake_mid_point - fake_red_tol), (fake_mid_point + fake_red_tol))
    )

    if pytest.gen.randint(0, 1) == 1:
        arg_val = (
            pytest.gen.randint(fake_yellow_tol, fake_red_tol - 1) + fake_mid_point
        )  # random val within upper red range
    else:
        arg_val = (
            pytest.gen.randint(0 - fake_red_tol, 0 - fake_yellow_tol - 1)
            + fake_mid_point
        )  # sometimes flip to lower red range
    if arg_val > 0:
        arg_val += pytest.gen.random()  # make float by adding some random decimal
    else:
        arg_val -= pytest.gen.random()  # make float by adding some random decimal

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    # Act
    result = cut.state(arg_val, arg_test_params, arg_epsilon)

    # Assert
    assert result == ("RED", [({"RED"}, 1.0)])


def test_TelemetryTestSuite_state_returns_tuple_of_str_3_dashes_and_list_containing_tuple_of_set_of_str_RED_YELLOW_and_GREEN_and_1_pt_0_when_int_val_is_in_not_in_any_range():
    # Arrange
    arg_test_params = []
    arg_epsilon = MagicMock()

    factor = 1
    if pytest.gen.randint(0, 1) == 1:
        factor *= -1
    # arbitrary, from 0 to 200 with 50/50 change of negative
    fake_mid_point = (
        pytest.gen.randint(0, 200) * factor
    )  # arbitrary, from 0 to 200 with 50/50 change of negative
    fake_green_tol = pytest.gen.randint(
        1, 50
    )  # arbitrary, from 1 to 50 allowance in both directions from fake_mid_point
    fake_yellow_tol = (
        pytest.gen.randint(1, 20) + fake_green_tol
    )  # arbitrary, from 1 to 20 allowance in both directions from fake_mid_point + fake_green_tol
    fake_red_tol = (
        pytest.gen.randint(1, 10) + fake_yellow_tol
    )  # arbitrary, from 1 to 10 allowance in both directions from fake_mid_point + fake_yellow_tol

    arg_test_params.append(
        range((fake_mid_point - fake_green_tol), (fake_mid_point + fake_green_tol))
    )
    arg_test_params.append(
        range((fake_mid_point - fake_yellow_tol), (fake_mid_point + fake_yellow_tol))
    )
    arg_test_params.append(
        range((fake_mid_point - fake_red_tol), (fake_mid_point + fake_red_tol))
    )

    if pytest.gen.randint(0, 1) == 1:
        arg_val = fake_red_tol + fake_mid_point + 1  # random val outside upper red
    else:
        arg_val = 0 - fake_red_tol + fake_mid_point - 1  # random val outside lower red
    if arg_val > 0:
        arg_val += pytest.gen.random()  # make float by adding some random decimal
    else:
        arg_val -= pytest.gen.random()  # make float by adding some random decimal

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    # Act
    result = cut.state(arg_val, arg_test_params, arg_epsilon)

    # Assert
    assert result == ("---", [({"RED", "YELLOW", "GREEN"}, 1.0)])


# feasibility tests
def test_TelemetryTestSuite_feasibility_asserts_len_given_test_params_is_not_2_or_4(
    mocker,
):
    # Arrange
    arg_val = MagicMock()
    arg_test_params = []
    arg_epsilon = MagicMock()

    num_test_params = pytest.gen.sample([1, 3, 5], 1)

    for i in range(num_test_params[0]):
        arg_test_params.append(MagicMock())

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    # Act
    with pytest.raises(AssertionError) as e_info:
        result = cut.feasibility(arg_val, arg_test_params, arg_epsilon)

    # Assert
    assert e_info.match("")


def test_TelemetryTestSuite_feasibility_return_expected_stat_and_mass_assignments_for_lower_boundry_val_eq_to_lowest_bound_when_given_test_params_length_2(
    mocker,
):
    # Arrange
    arg_epsilon = MagicMock()

    fake_lowest_bound = pytest.gen.randint(-100, 100)  # arbitrary, from -100 to 100
    fake_highest_bound = fake_lowest_bound + pytest.gen.randint(
        10, 100
    )  # arbitrary, from 10 to 100 higher than lowest bound

    arg_test_params = [fake_lowest_bound, fake_highest_bound]
    arg_val = fake_lowest_bound

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    # Act
    result = cut.feasibility(arg_val, arg_test_params, arg_epsilon)

    # Assert
    assert result == ("RED", [({"RED", "GREEN"}, 1.0)])


def test_TelemetryTestSuite_feasibility_return_expected_stat_and_mass_assignments_for_lower_boundry_given_val_eq_to_lowest_bound_when_given_test_params_length_4(
    mocker,
):
    # Arrange
    arg_epsilon = MagicMock()

    fake_lowest_bound = pytest.gen.randint(-100, 100)  # arbitrary, from -100 to 100
    fake_highest_bound = fake_lowest_bound + pytest.gen.randint(
        10, 100
    )  # arbitrary, from 10 to 100 higher than lowest bound

    arg_test_params = [
        fake_lowest_bound,
        fake_lowest_bound + 1,
        fake_highest_bound - 1,
        fake_highest_bound,
    ]

    arg_val = fake_lowest_bound

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    # Act
    result = cut.feasibility(arg_val, arg_test_params, arg_epsilon)

    # Assert
    assert result == ("RED", [({"RED", "YELLOW"}, 1.0)])


def test_TelemetryTestSuite_feasibility_return_expected_stat_and_mass_assignments_for_lower_boundry_given_val_less_than_low_range_minus_delta_when_given_test_params_length_2(
    mocker,
):
    # Arrange
    arg_epsilon = 1.0

    fake_lowest_bound = pytest.gen.randint(-100, 100)  # arbitrary, from -100 to 100
    fake_highest_bound = fake_lowest_bound + pytest.gen.randint(
        10, 100
    )  # arbitrary, from 10 to 100 higher than lowest bound
    fake_delta = arg_epsilon * abs(fake_highest_bound - fake_lowest_bound)

    arg_test_params = [fake_lowest_bound, fake_highest_bound]

    arg_val = fake_lowest_bound - fake_delta - 1  # -1 for less than low minus delta

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    # Act
    result = cut.feasibility(arg_val, arg_test_params, arg_epsilon)

    # Assert
    assert result == ("RED", [({"RED"}, 1.0)])


def test_TelemetryTestSuite_feasibility_return_expected_stat_and_mass_assignments_for_lower_boundry_given_val_less_than_low_range_minus_delta_when_given_test_params_length_4(
    mocker,
):
    # Arrange
    arg_epsilon = 1.0

    fake_lowest_bound = pytest.gen.randint(-100, 100)  # arbitrary, from -100 to 100
    fake_highest_bound = fake_lowest_bound + pytest.gen.randint(
        10, 100
    )  # arbitrary, from 10 to 100 higher than lowest bound
    fake_delta = arg_epsilon * abs(fake_highest_bound - fake_lowest_bound)

    arg_test_params = [fake_lowest_bound, fake_highest_bound]

    arg_val = fake_lowest_bound - fake_delta - 1  # -1 for less than low minus delta

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    # Act
    result = cut.feasibility(arg_val, arg_test_params, arg_epsilon)

    # Assert
    assert result == ("RED", [({"RED"}, 1.0)])


def test_TelemetryTestSuite_feasibility_return_expected_stat_and_mass_assignments_for_lower_boundry_given_val_within_low_range_minus_delta_when_given_test_params_length_2(
    mocker,
):
    # Arrange
    arg_epsilon = 1.0

    fake_lowest_bound = pytest.gen.randint(-100, 100)  # arbitrary, from -100 to 100
    fake_highest_bound = fake_lowest_bound + pytest.gen.randint(
        10, 100
    )  # arbitrary, from 10 to 100 higher than lowest bound
    fake_delta = arg_epsilon * abs(fake_highest_bound - fake_lowest_bound)

    arg_test_params = [fake_lowest_bound, fake_highest_bound]
    arg_val = fake_lowest_bound - 1

    expected_mass = abs(fake_lowest_bound - arg_val) / fake_delta
    expected_red_yellow_mass = 1.0 - expected_mass

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    # Act
    result = cut.feasibility(arg_val, arg_test_params, arg_epsilon)

    # Assert
    assert result == (
        "RED",
        [({"RED"}, expected_mass), ({"GREEN", "RED"}, expected_red_yellow_mass)],
    )


def test_TelemetryTestSuite_feasibility_return_expected_stat_and_mass_assignments_for_lower_boundry_given_val_within_low_range_minus_delta_when_given_test_params_length_4(
    mocker,
):
    # Arrange
    arg_epsilon = 1.0

    fake_lowest_bound = pytest.gen.randint(-100, 100)  # arbitrary, from -100 to 100
    fake_highest_bound = fake_lowest_bound + pytest.gen.randint(
        10, 100
    )  # arbitrary, from 10 to 100 higher than lowest bound
    fake_delta = arg_epsilon * abs(fake_highest_bound - fake_highest_bound - 2)

    arg_test_params = [
        fake_lowest_bound,
        fake_lowest_bound + 2,
        fake_highest_bound - 2,
        fake_highest_bound,
    ]
    arg_val = fake_lowest_bound - 1

    expected_mass = abs(fake_lowest_bound - arg_val) / fake_delta
    expected_red_yellow_mass = 1.0 - expected_mass

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    # Act
    result = cut.feasibility(arg_val, arg_test_params, arg_epsilon)

    # Assert
    assert result == (
        "RED",
        [({"RED"}, expected_mass), ({"YELLOW", "RED"}, expected_red_yellow_mass)],
    )


def test_TelemetryTestSuite_feasibility_return_expected_stat_and_mass_assignments_for_high_boundry_val_eq_to_lowest_bound_when_given_test_params_length_2(
    mocker,
):
    # Arrange
    arg_epsilon = MagicMock()

    fake_lowest_bound = pytest.gen.randint(-100, 100)  # arbitrary, from -100 to 100
    fake_highest_bound = fake_lowest_bound + pytest.gen.randint(
        10, 100
    )  # arbitrary, from 10 to 100 higher than lowest bound

    arg_test_params = [fake_lowest_bound, fake_highest_bound]
    arg_val = fake_highest_bound

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    # Act
    result = cut.feasibility(arg_val, arg_test_params, arg_epsilon)

    # Assert
    assert result == ("RED", [({"RED", "GREEN"}, 1.0)])


def test_TelemetryTestSuite_feasibility_return_expected_stat_and_mass_assignments_for_high_boundry_given_val_eq_to_lowest_bound_when_given_test_params_length_4(
    mocker,
):
    # Arrange
    arg_epsilon = MagicMock()

    fake_lowest_bound = pytest.gen.randint(-100, 100)  # arbitrary, from -100 to 100
    fake_highest_bound = fake_lowest_bound + pytest.gen.randint(
        10, 100
    )  # arbitrary, from 10 to 100 higher than lowest bound

    arg_test_params = [
        fake_lowest_bound,
        fake_lowest_bound + 1,
        fake_highest_bound - 1,
        fake_highest_bound,
    ]

    arg_val = fake_highest_bound

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    # Act
    result = cut.feasibility(arg_val, arg_test_params, arg_epsilon)

    # Assert
    assert result == ("RED", [({"RED", "YELLOW"}, 1.0)])


def test_TelemetryTestSuite_feasibility_return_expected_stat_and_mass_assignments_for_high_boundry_given_val_less_than_low_range_minus_delta_when_given_test_params_length_2(
    mocker,
):
    # Arrange
    arg_epsilon = 1.0

    fake_lowest_bound = pytest.gen.randint(-100, 100)  # arbitrary, from -100 to 100
    fake_highest_bound = fake_lowest_bound + pytest.gen.randint(
        10, 100
    )  # arbitrary, from 10 to 100 higher than lowest bound
    fake_delta = arg_epsilon * abs(fake_highest_bound - fake_lowest_bound)

    arg_test_params = [fake_lowest_bound, fake_highest_bound]

    arg_val = fake_highest_bound + fake_delta + 1  # +1 for more than high plus delta

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    # Act
    result = cut.feasibility(arg_val, arg_test_params, arg_epsilon)

    # Assert
    assert result == ("RED", [({"RED"}, 1.0)])


def test_TelemetryTestSuite_feasibility_return_expected_stat_and_mass_assignments_for_high_boundry_given_val_less_than_low_range_minus_delta_when_given_test_params_length_4(
    mocker,
):
    # Arrange
    arg_epsilon = 1.0

    fake_lowest_bound = pytest.gen.randint(-100, 100)  # arbitrary, from -100 to 100
    fake_highest_bound = fake_lowest_bound + pytest.gen.randint(
        10, 100
    )  # arbitrary, from 10 to 100 higher than lowest bound
    fake_delta = arg_epsilon * abs(fake_highest_bound - fake_lowest_bound)

    arg_test_params = [fake_lowest_bound, fake_highest_bound]

    arg_val = fake_highest_bound + fake_delta + 1  # +1 for more than high plus delta

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    # Act
    result = cut.feasibility(arg_val, arg_test_params, arg_epsilon)

    # Assert
    assert result == ("RED", [({"RED"}, 1.0)])


def test_TelemetryTestSuite_feasibility_return_expected_stat_and_mass_assignments_for_high_boundry_given_val_within_low_range_minus_delta_when_given_test_params_length_2(
    mocker,
):
    # Arrange
    arg_epsilon = 1.0

    fake_lowest_bound = pytest.gen.randint(-100, 100)  # arbitrary, from -100 to 100
    fake_highest_bound = fake_lowest_bound + pytest.gen.randint(
        10, 100
    )  # arbitrary, from 10 to 100 higher than lowest bound
    fake_delta = arg_epsilon * abs(fake_highest_bound - fake_lowest_bound)

    arg_test_params = [fake_lowest_bound, fake_highest_bound]
    arg_val = fake_highest_bound + 1

    expected_mass = abs(fake_highest_bound - arg_val) / fake_delta
    expected_red_yellow_mass = 1.0 - expected_mass

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    # Act
    result = cut.feasibility(arg_val, arg_test_params, arg_epsilon)

    # Assert
    assert result == (
        "RED",
        [({"RED"}, expected_mass), ({"RED", "GREEN"}, expected_red_yellow_mass)],
    )


def test_TelemetryTestSuite_feasibility_return_expected_stat_and_mass_assignments_for_high_boundry_given_val_within_low_range_minus_delta_when_given_test_params_length_4(
    mocker,
):
    # Arrange
    arg_epsilon = 1.0

    fake_lowest_bound = pytest.gen.randint(-100, 100)  # arbitrary, from -100 to 100
    fake_highest_bound = fake_lowest_bound + pytest.gen.randint(
        10, 100
    )  # arbitrary, from 10 to 100 higher than lowest bound
    fake_delta = arg_epsilon * abs(fake_highest_bound - fake_highest_bound - 2)

    arg_test_params = [
        fake_lowest_bound,
        fake_lowest_bound + 2,
        fake_highest_bound - 2,
        fake_highest_bound,
    ]
    arg_val = fake_highest_bound + 1

    expected_mass = abs(fake_highest_bound - arg_val) / fake_delta
    expected_red_yellow_mass = 1.0 - expected_mass

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    # Act
    result = cut.feasibility(arg_val, arg_test_params, arg_epsilon)

    # Assert
    assert result == (
        "RED",
        [({"RED"}, expected_mass), ({"YELLOW", "RED"}, expected_red_yellow_mass)],
    )


def test_TelemetryTestSuite_feasibility_return_expected_stat_and_mass_assignments_for_within_bound_val_in_green_zone_when_given_test_params_length_2(
    mocker,
):
    # Arrange
    arg_epsilon = 1.0

    fake_lowest_bound = pytest.gen.randint(-100, 100)  # arbitrary, from -100 to 100
    fake_highest_bound = fake_lowest_bound + pytest.gen.randint(
        10, 100
    )  # arbitrary, from 10 to 100 higher than lowest bound

    arg_test_params = [fake_lowest_bound, fake_highest_bound]
    arg_val = int(
        fake_highest_bound - (abs(fake_highest_bound - fake_lowest_bound) / 2)
    )

    fake_delta = arg_epsilon * (abs(fake_highest_bound - fake_lowest_bound))
    expected_mass = abs(fake_lowest_bound - arg_val) / fake_delta
    print(fake_lowest_bound, arg_val, fake_highest_bound)

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    # Act
    result = cut.feasibility(arg_val, arg_test_params, arg_epsilon)

    # Assert
    assert result == (
        "GREEN",
        [({"GREEN"}, expected_mass), ({"GREEN", "RED"}, 1.0 - expected_mass)],
    )


def test_TelemetryTestSuite_feasibility_return_expected_stat_and_mass_assignments_for_within_bound_val_in_green_zone_when_given_test_params_length_4(
    mocker,
):
    # Arrange
    arg_epsilon = 1.0

    fake_lowest_bound = pytest.gen.randint(-100, 100)  # arbitrary, from -100 to 100
    fake_highest_bound = fake_lowest_bound + pytest.gen.randint(
        10, 100
    )  # arbitrary, from 3 to 100 higher than lowest bound

    arg_test_params = [
        fake_lowest_bound,
        fake_lowest_bound + 1,
        fake_highest_bound - 1,
        fake_highest_bound,
    ]

    fake_delta = arg_epsilon * (abs(fake_highest_bound - fake_lowest_bound))

    arg_val = int(
        fake_highest_bound - (abs(fake_highest_bound - fake_lowest_bound) / 2)
    )
    print(
        fake_lowest_bound,
        arg_val,
        fake_lowest_bound + 2,
        fake_highest_bound - 2,
        fake_highest_bound,
    )

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    # Act
    result = cut.feasibility(arg_val, arg_test_params, arg_epsilon)

    # Assert
    assert result == ("GREEN", [({"GREEN"}, 1.0)])


def test_TelemetryTestSuite_feasibility_return_expected_stat_and_mass_assignments_for_within_bound_val_in_yellow_low_zone_when_given_test_params_length_4(
    mocker,
):
    # Arrange
    arg_epsilon = 1.0

    fake_lowest_bound = pytest.gen.randint(-100, 100)  # arbitrary, from -100 to 100
    fake_highest_bound = fake_lowest_bound + pytest.gen.randint(
        10, 100
    )  # arbitrary, from 10 to 100 higher than lowest bound

    arg_test_params = [
        fake_lowest_bound,
        fake_lowest_bound + 2,
        fake_highest_bound - 2,
        fake_highest_bound,
    ]

    fake_delta = arg_epsilon * (abs(fake_highest_bound - fake_lowest_bound))

    arg_val = fake_lowest_bound + 1
    print(
        fake_lowest_bound,
        arg_val,
        fake_lowest_bound + 2,
        fake_highest_bound - 2,
        fake_highest_bound,
    )

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    # Act
    result = cut.feasibility(arg_val, arg_test_params, arg_epsilon)

    # Assert
    assert result == ("YELLOW", [({"YELLOW"}, 0.5), ({"RED", "YELLOW"}, 0.5)])


def test_TelemetryTestSuite_feasibility_return_expected_stat_and_mass_assignments_for_within_bound_val_in_yellow_high_zone_when_given_test_params_length_4(
    mocker,
):
    # Arrange
    arg_epsilon = 0.5

    fake_lowest_bound = pytest.gen.randint(-100, 100)  # arbitrary, from -100 to 100
    fake_highest_bound = fake_lowest_bound + pytest.gen.randint(
        20, 100
    )  # arbitrary, from 10 to 100 higher than lowest bound

    arg_test_params = [
        fake_lowest_bound,
        fake_lowest_bound + 4,
        fake_highest_bound - 4,
        fake_highest_bound,
    ]

    arg_val = fake_highest_bound - 1
    print(
        fake_lowest_bound,
        fake_lowest_bound + 4,
        fake_highest_bound - 4,
        arg_val,
        fake_highest_bound,
    )
    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    # Act
    result = cut.feasibility(arg_val, arg_test_params, arg_epsilon)

    # Assert
    assert result == ("YELLOW", [({"YELLOW"}, 0.5), ({"YELLOW", "RED"}, 0.5)])


def test_TelemetryTestSuite_feasibility_return_expected_stat_and_mass_assignments_for_within_bound_val_on_yellow_high_mark_when_given_test_params_length_4(
    mocker,
):
    # Arrange
    arg_epsilon = 0.5

    fake_lowest_bound = pytest.gen.randint(-100, 100)  # arbitrary, from -100 to 100
    fake_highest_bound = fake_lowest_bound + pytest.gen.randint(
        20, 100
    )  # arbitrary, from 10 to 100 higher than lowest bound

    arg_test_params = [
        fake_lowest_bound,
        fake_lowest_bound + 4,
        fake_highest_bound - 4,
        fake_highest_bound,
    ]

    arg_val = fake_highest_bound - 4
    print(
        fake_lowest_bound,
        fake_lowest_bound + 4,
        fake_highest_bound - 4,
        arg_val,
        fake_highest_bound,
    )
    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    # Act
    result = cut.feasibility(arg_val, arg_test_params, arg_epsilon)

    # Assert
    assert result == ("YELLOW", [({"YELLOW", "GREEN"}, 1.0)])


# noop tests
def test_TelemetryTestSuite_noop_returns_tuple_of_str_GREEN_and_list_containing_tuple_of_set_of_str_GREEN_and_1_pt_0():
    # Arrange
    arg_val = MagicMock()
    arg_test_params = MagicMock()
    arg_epsilon = MagicMock()

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    # Act
    result = cut.noop(arg_val, arg_test_params, arg_epsilon)

    # Assert
    assert result == ("GREEN", [({"GREEN"}, 1.0)])


# calc_single_status tests
def test_TelemetryTestSuite_calc_single_status_returns_tuple_of_value_from_call_to_most_common_on_occurrences_and_1_pt_0_when_mode_is_not_str_max_or_str_distr_or_str_strict(
    mocker,
):
    # Arrange
    arg_status_list = MagicMock()
    arg_mode = MagicMock()

    fake_occurrences = MagicMock()
    fake_max_occurrence = MagicMock()

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    mocker.patch(
        telemetry_test_suite.__name__ + ".Counter", return_value=fake_occurrences
    )
    mocker.patch.object(
        fake_occurrences, "most_common", return_value=[[fake_max_occurrence]]
    )

    # Act
    result = cut.calc_single_status(arg_status_list, arg_mode)

    # Assert
    assert telemetry_test_suite.Counter.call_count == 1
    assert telemetry_test_suite.Counter.call_args_list[0].args == (arg_status_list,)
    assert result == (fake_max_occurrence, 1.0)


def test_TelemetryTestSuite_calc_single_status_returns_tuple_of_value_from_call_to_most_common_on_occurrences_and_1_pt_0_when_mode_is_str_max(
    mocker,
):
    # Arrange
    arg_status_list = MagicMock()
    arg_mode = "max"

    fake_occurrences = MagicMock()
    fake_max_occurrence = MagicMock()

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    mocker.patch(
        telemetry_test_suite.__name__ + ".Counter", return_value=fake_occurrences
    )
    mocker.patch.object(
        fake_occurrences, "most_common", return_value=[[fake_max_occurrence]]
    )

    # Act
    result = cut.calc_single_status(arg_status_list, arg_mode)

    # Assert
    assert telemetry_test_suite.Counter.call_count == 1
    assert telemetry_test_suite.Counter.call_args_list[0].args == (arg_status_list,)
    assert result == (fake_max_occurrence, 1.0)


def test_TelemetryTestSuite_calc_single_status_returns_tuple_of_value_from_call_to_most_common_on_occurrences_and_ratio_of_max_occurrence_over_len_given_status_list_when_mode_is_str_distr(
    mocker,
):
    # Arrange
    arg_status_list = []
    arg_mode = "distr"

    num_fake_statuses = pytest.gen.randint(
        1, 10
    )  # arbitrary, from 1 to 10 (0 not allowed, div by 0 error)
    fake_max_occurrence = MagicMock()

    for i in range(num_fake_statuses):
        arg_status_list.append(MagicMock())
    fake_occurrences = telemetry_test_suite.Counter.__new__(
        telemetry_test_suite.Counter
    )

    expected_float = fake_occurrences[fake_max_occurrence] / num_fake_statuses

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    mocker.patch(
        telemetry_test_suite.__name__ + ".Counter", return_value=fake_occurrences
    )
    mocker.patch.object(
        fake_occurrences, "most_common", return_value=[[fake_max_occurrence]]
    )

    # Act
    result = cut.calc_single_status(arg_status_list, arg_mode)

    # Assert
    assert telemetry_test_suite.Counter.call_count == 1
    assert telemetry_test_suite.Counter.call_args_list[0].args == (arg_status_list,)
    assert result == (fake_max_occurrence, expected_float)


def test_TelemetryTestSuite_calc_single_status_returns_tuple_of_value_from_call_to_most_common_on_occurrences_and_1_pt_0_when_mode_is_str_strict_and_no_occurrences_of_str_RED(
    mocker,
):
    # Arrange
    arg_status_list = []
    arg_mode = "strict"

    num_fake_statuses = pytest.gen.randint(
        1, 10
    )  # arbitrary, from 1 to 10 (0 not allowed, div by 0 error)
    fake_max_occurrence = MagicMock()

    for i in range(num_fake_statuses):
        arg_status_list.append(MagicMock())
    fake_occurrences = telemetry_test_suite.Counter.__new__(
        telemetry_test_suite.Counter
    )

    expected_float = fake_occurrences[fake_max_occurrence] / num_fake_statuses

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    mocker.patch(
        telemetry_test_suite.__name__ + ".Counter", return_value=fake_occurrences
    )
    mocker.patch.object(
        fake_occurrences, "most_common", return_value=[[fake_max_occurrence]]
    )

    # Act
    result = cut.calc_single_status(arg_status_list, arg_mode)

    # Assert
    assert telemetry_test_suite.Counter.call_count == 1
    assert telemetry_test_suite.Counter.call_args_list[0].args == (arg_status_list,)
    assert result == (fake_max_occurrence, 1.0)


def test_TelemetryTestSuite_calc_single_status_returns_tuple_of_str_RED_and_1_pt_0_when_mode_is_str_strict_and_ratio_of_RED_occurrence_over_len_given_status_list_with_occurrences_of_str_RED(
    mocker,
):
    # Arrange
    arg_status_list = []
    arg_mode = "strict"

    num_fake_statuses = pytest.gen.randint(
        1, 10
    )  # arbitrary, from 1 to 10 (0 not allowed, div by 0 error)
    num_red_statuses = pytest.gen.randint(
        1, num_fake_statuses
    )  # arbitrary, from 1 to total statuses
    fake_max_occurrence = MagicMock()

    for i in range(num_fake_statuses):
        arg_status_list.append(MagicMock())

    fake_occurrences = telemetry_test_suite.Counter(["RED"] * num_red_statuses)

    expected_float = fake_occurrences["RED"] / num_fake_statuses

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    mocker.patch(
        telemetry_test_suite.__name__ + ".Counter", return_value=fake_occurrences
    )
    mocker.patch.object(
        fake_occurrences, "most_common", return_value=[[fake_max_occurrence]]
    )

    # Act
    result = cut.calc_single_status(arg_status_list, arg_mode)

    # Assert
    assert telemetry_test_suite.Counter.call_count == 1
    assert telemetry_test_suite.Counter.call_args_list[0].args == (arg_status_list,)
    assert result == ("RED", expected_float)


def test_TelemetryTestSuite_calc_single_status_default_given_mode_is_str_strict(mocker):
    # Arrange
    arg_status_list = []

    num_fake_statuses = pytest.gen.randint(
        1, 10
    )  # arbitrary, from 1 to 10 (0 not allowed, div by 0 error)
    num_red_statuses = pytest.gen.randint(
        1, num_fake_statuses
    )  # arbitrary, from 1 to total statuses
    fake_max_occurrence = MagicMock()

    for i in range(num_fake_statuses):
        arg_status_list.append(MagicMock())

    fake_occurrences = telemetry_test_suite.Counter(["RED"] * num_red_statuses)

    expected_float = fake_occurrences["RED"] / num_fake_statuses

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)

    mocker.patch(
        telemetry_test_suite.__name__ + ".Counter", return_value=fake_occurrences
    )
    mocker.patch.object(
        fake_occurrences, "most_common", return_value=[[fake_max_occurrence]]
    )

    # Act
    result = cut.calc_single_status(arg_status_list)

    # Assert
    assert telemetry_test_suite.Counter.call_count == 1
    assert telemetry_test_suite.Counter.call_args_list[0].args == (arg_status_list,)
    assert result == ("RED", expected_float)


# get_suite_status
def test_TelemetryTestSuite_get_suite_status_raises_TypeError_when_latest_results_is_None():
    # Arrange
    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)
    cut.latest_results = None

    # Act
    with pytest.raises(TypeError) as e_info:
        result = cut.get_suite_status()

    # Assert
    assert e_info.match("'NoneType' object is not iterable")


def test_TelemetryTestSuite_get_suite_status_returns_value_from_call_to_calc_single_status_when_it_is_given_empty_list_because_latest_results_are_empty(
    mocker,
):
    # Arrange
    expected_result = MagicMock()

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)
    cut.latest_results = []

    mocker.patch.object(cut, "calc_single_status", return_value=expected_result)

    # Act
    result = cut.get_suite_status()

    # Assert
    assert cut.calc_single_status.call_count == 1
    assert cut.calc_single_status.call_args_list[0].args == ([],)
    assert result == expected_result


def test_TelemetryTestSuite_get_suite_status_returns_value_from_call_to_calc_single_status_when_it_is_given_list_of_all_statuses_in_latest_results(
    mocker,
):
    # Arrange
    num_fake_results = pytest.gen.randint(
        1, 10
    )  # arbitrary, from 1 to 10 (0 has its own test)
    fake_latest_results = []
    fake_statuses = []

    for i in range(num_fake_results):
        fake_res = MagicMock()
        fake_status = MagicMock()

        mocker.patch.object(fake_res, "get_status", return_value=fake_status)

        fake_latest_results.append(fake_res)
        fake_statuses.append(fake_status)

    expected_result = MagicMock()

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)
    cut.latest_results = fake_latest_results

    mocker.patch.object(cut, "calc_single_status", return_value=expected_result)

    # Act
    result = cut.get_suite_status()

    # Assert
    assert cut.calc_single_status.call_count == 1
    assert cut.calc_single_status.call_args_list[0].args == (fake_statuses,)
    assert result == expected_result


# get_status_specific_mnemonics
# test_get_status_specific_mnemonics_raises_TypeError_when_latest_results_is_None was written because None is the init value for latest_results
def test_TelemetryTestSuite_get_status_specific_mnemonics_raises_TypeError_when_latest_results_is_None(
    mocker,
):
    # Arrange
    arg_status = MagicMock()

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)
    cut.latest_results = None

    # Act
    with pytest.raises(TypeError) as e_info:
        result = cut.get_status_specific_mnemonics(arg_status)

    # Assert
    assert e_info.match("'NoneType' object is not iterable")


def test_TelemetryTestSuite_get_status_specific_mnemonics_returns_empty_list_when_latest_results_is_empty(
    mocker,
):
    # Arrange
    arg_status = MagicMock()

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)
    cut.latest_results = []

    # Act
    result = cut.get_status_specific_mnemonics(arg_status)

    # Assert
    assert result == []


def test_TelemetryTestSuite_get_status_specific_mnemonics_returns_the_only_name_in_latest_results_because_its_status_eq_given_status(
    mocker,
):
    # Arrange
    arg_status = MagicMock()

    fake_res = MagicMock()

    expected_name = str(MagicMock())

    mocker.patch.object(fake_res, "get_status", return_value=arg_status)
    mocker.patch.object(fake_res, "get_name", return_value=expected_name)

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)
    cut.latest_results = [fake_res]

    # Act
    result = cut.get_status_specific_mnemonics(arg_status)

    # Assert
    assert result == [expected_name]


def test_TelemetryTestSuite_get_status_specific_mnemonics_returns_empty_list_latest_results_because_its_status_not_eq_given_status(
    mocker,
):
    # Arrange
    arg_status = MagicMock()

    fake_res = MagicMock()

    mocker.patch.object(fake_res, "get_status", return_value=MagicMock())

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)
    cut.latest_results = [fake_res]

    # Act
    result = cut.get_status_specific_mnemonics(arg_status)

    # Assert
    assert result == []


def test_TelemetryTestSuite_get_status_specific_mnemonics_returns_only_names_in_latest_results_where_status_matches_given_status(
    mocker,
):
    # Arrange
    arg_status = MagicMock()

    num_fake_results = pytest.gen.randint(
        2, 10
    )  # arbitrary, from 2 to 10 (0 and 1 both have own test)
    num_fake_status_matches = pytest.gen.randint(
        1, num_fake_results - 1
    )  # at least 1 match up to 1 less than all
    fake_latest_results = [False] * num_fake_results

    expected_names = []

    for i in pytest.gen.sample(
        range(len(fake_latest_results)), num_fake_status_matches
    ):
        fake_latest_results[i] = True

    for i in range(len(fake_latest_results)):
        fake_res = MagicMock()
        if fake_latest_results[i] == True:
            fake_name = str(MagicMock())
            mocker.patch.object(fake_res, "get_status", return_value=arg_status)
            mocker.patch.object(fake_res, "get_name", return_value=fake_name)
            expected_names.append(fake_name)
        else:
            mocker.patch.object(fake_res, "get_status", return_value=MagicMock())
        fake_latest_results[i] = fake_res

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)
    cut.latest_results = fake_latest_results

    # Act
    result = cut.get_status_specific_mnemonics(arg_status)

    # Assert
    assert result == expected_names
    assert len(result) != len(fake_latest_results)


def test_TelemetryTestSuite_get_status_specific_mnemonics_returns_all_names_in_latest_results_when_all_statuses_matches_given_status(
    mocker,
):
    # Arrange
    arg_status = MagicMock()

    num_fake_results = pytest.gen.randint(
        1, 10
    )  # arbitrary, from 2 to 10 (0 and 1 both have own test)
    fake_latest_results = []

    expected_names = []

    for i in range(num_fake_results):
        fake_res = MagicMock()
        fake_name = str(MagicMock())
        mocker.patch.object(fake_res, "get_status", return_value=arg_status)
        mocker.patch.object(fake_res, "get_name", return_value=fake_name)
        fake_latest_results.append(fake_res)
        expected_names.append(fake_name)

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)
    cut.latest_results = fake_latest_results

    # Act
    result = cut.get_status_specific_mnemonics(arg_status)

    # Assert
    assert result == expected_names
    assert len(result) == len(fake_latest_results)


def test_TelemetryTestSuite_get_status_specific_mnemonics_default_given_status_is_str_RED(
    mocker,
):
    # Arrange
    fake_res = MagicMock()

    expected_name = str(MagicMock())

    mocker.patch.object(fake_res, "get_status", return_value="RED")
    mocker.patch.object(fake_res, "get_name", return_value=expected_name)

    cut = TelemetryTestSuite.__new__(TelemetryTestSuite)
    cut.latest_results = [fake_res]

    # Act
    result = cut.get_status_specific_mnemonics()

    # Assert
    assert result == [expected_name]
