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
from unittest.mock import MagicMock

import onair.src.systems.vehicle_rep as vehicle_rep
from onair.src.systems.vehicle_rep import VehicleRepresentation


# __init__ tests
def test_VehicleRepresentation__init__asserts_when_len_given_headers_is_not_eq_to_len_given_tests(
    mocker,
):
    # Arrange
    arg_headers = MagicMock()
    arg_tests = MagicMock()

    fake_len = []
    fake_len.append(pytest.gen.randint(0, 100))  # arbitrary, from 0 to 100 size
    fake_len.append(fake_len[0])
    while fake_len[1] == fake_len[0]:  # need a value not equal for test to pass
        fake_len[1] = pytest.gen.randint(0, 100)  # arbitrary, same as fake_len_headers

    cut = VehicleRepresentation.__new__(VehicleRepresentation)

    mocker.patch(vehicle_rep.__name__ + ".len", side_effect=fake_len)
    # Act
    with pytest.raises(AssertionError) as e_info:
        cut.__init__(arg_headers, arg_tests)

    # Assert
    assert vehicle_rep.len.call_count == 2
    call_list = set({})
    [
        call_list.add(vehicle_rep.len.call_args_list[i].args)
        for i in range(len(vehicle_rep.len.call_args_list))
    ]
    assert call_list == {(arg_headers,), (arg_tests,)}
    assert e_info.match("")


def test_VehicleRepresentation__init__sets_status_to_Status_with_str_MISSION_and_headers_to_given_headers_and_test_suite_to_TelemetryTestSuite_with_given_headers_and_tests_and_curr_data_to_all_empty_step_len_of_headers(
    mocker,
):
    # Arrange
    arg_headers = MagicMock()
    arg_tests = MagicMock()

    fake_len = pytest.gen.randint(0, 100)  # arbitrary, 0 to 100 items
    fake_status = MagicMock()
    fake_test_suite = MagicMock()

    cut = VehicleRepresentation.__new__(VehicleRepresentation)

    mocker.patch(vehicle_rep.__name__ + ".len", return_value=fake_len)
    mocker.patch(vehicle_rep.__name__ + ".Status", return_value=fake_status)
    mocker.patch(
        vehicle_rep.__name__ + ".TelemetryTestSuite", return_value=fake_test_suite
    )

    # Act
    cut.__init__(arg_headers, arg_tests)

    # Assert
    assert vehicle_rep.Status.call_count == 1
    assert vehicle_rep.Status.call_args_list[0].args == ("MISSION",)
    assert cut.status == fake_status
    assert cut.headers == arg_headers
    assert vehicle_rep.TelemetryTestSuite.call_count == 1
    assert vehicle_rep.TelemetryTestSuite.call_args_list[0].args == (
        arg_headers,
        arg_tests,
    )
    assert cut.test_suite == fake_test_suite
    assert cut.curr_data == ["-"] * fake_len


# update tests
def test_VehicleRepresentation_update_calls_update_constructs_then_update_curr_data_then_executes_test_suite_and_finally_sets_status(
    mocker,
):
    # Arrange
    mock_manager = mocker.MagicMock()
    arg_frame = MagicMock()

    fake_suite_status = []
    num_fake_status = pytest.gen.randint(0, 10)  # from 0 to 10 arbitrary
    for i in range(num_fake_status):
        fake_suite_status.append(MagicMock())

    cut = VehicleRepresentation.__new__(VehicleRepresentation)
    cut.test_suite = MagicMock()
    cut.curr_data = MagicMock()
    cut.status = MagicMock()

    mock_manager.attach_mock(
        mocker.patch.object(cut, "update_constructs"), "update_constructs"
    )
    mock_manager.attach_mock(
        mocker.patch.object(cut, "update_curr_data"), "update_curr_data"
    )
    mock_manager.attach_mock(
        mocker.patch.object(cut.test_suite, "execute_suite"), "test_suite.execute_suite"
    )
    mock_manager.attach_mock(
        mocker.patch.object(
            cut.test_suite, "get_suite_status", return_value=fake_suite_status
        ),
        "test_suite.get_suite_status",
    )
    mock_manager.attach_mock(
        mocker.patch.object(cut.status, "set_status"), "status.set_status"
    )

    # Act
    cut.update(arg_frame)

    # Assert
    mock_manager.assert_has_calls(
        [
            mocker.call.update_curr_data(arg_frame),
            mocker.call.test_suite.execute_suite(cut.curr_data),
            mocker.call.test_suite.get_suite_status(),
            mocker.call.status.set_status(*fake_suite_status),
            mocker.call.update_constructs(cut.curr_data),
        ],
        any_order=False,
    )


# update_constructs tests
def test_VehicleRepresentation_update_constructs_does_nothing_when_knowledge_synthesis_constructs_are_empty(
    mocker,
):
    # Arrange
    arg_frame = MagicMock()

    fake_constructs = []

    cut = VehicleRepresentation.__new__(VehicleRepresentation)
    cut.knowledge_synthesis_constructs = fake_constructs

    # Act
    result = cut.update_constructs(arg_frame)

    # Assert
    assert result == None


def test_VehicleRepresentation_update_constructs_calls_update_on_each_knowledge_synthesis_construct(
    mocker,
):
    # Arrange
    arg_frame = MagicMock()

    num_fake_constructs = pytest.gen.randint(
        1, 10
    )  # from 1 to 10 arbitrary, 0 has own test
    fake_constructs = []

    for i in range(num_fake_constructs):
        fake_construct = MagicMock()
        fake_constructs.append(fake_construct)
        mocker.patch.object(fake_construct, "update")

    cut = VehicleRepresentation.__new__(VehicleRepresentation)
    cut.knowledge_synthesis_constructs = fake_constructs

    # Act
    result = cut.update_constructs(arg_frame)

    # Assert
    for i in range(num_fake_constructs):
        fake_constructs[i].update.call_count == 1
        fake_constructs[i].update.call_args_list[0].args == (arg_frame,)


# update_curr_data tests
def test_VehicleRepresentation_update_does_nothing_when_given_frame_is_empty(mocker):
    # Arrange
    arg_frame = MagicMock()
    arg_frame.__len__.return_value = 0

    cut = VehicleRepresentation.__new__(VehicleRepresentation)

    # Act
    result = cut.update_curr_data(arg_frame)

    # Assert
    assert result == None


def test_VehicleRepresentation_update_copies_all_frame_data_into_curr_data_when_all_frame_data_occupied(
    mocker,
):
    # Arrange
    arg_frame = []
    num_items_in_arg_frame = pytest.gen.randint(
        1, 10
    )  # from 1 to 10 arbitrary, 0 has own test

    fake_curr_data = []

    for i in range(num_items_in_arg_frame):
        arg_frame.append(MagicMock())
        fake_curr_data.append(MagicMock())
    assert fake_curr_data != arg_frame  # sanity check

    cut = VehicleRepresentation.__new__(VehicleRepresentation)
    cut.curr_data = fake_curr_data

    # Act
    cut.update_curr_data(arg_frame)

    # Assert
    assert cut.curr_data == arg_frame


def test_VehicleRepresentation_update_copies_only_occupied_frame_data_into_curr_data_when_some_frame_data_vacant(
    mocker,
):
    # Arrange
    arg_frame = []

    num_items_in_arg_frame = pytest.gen.randint(
        1, 10
    )  # from 1 to 10 arbitrary, 0 has own test
    fake_curr_data = []

    for i in range(num_items_in_arg_frame):
        arg_frame.append(MagicMock())
        fake_curr_data.append(MagicMock())
    assert fake_curr_data != arg_frame  # sanity check

    expected_curr_data = arg_frame.copy()
    num_vacant_frame_data = pytest.gen.randint(
        1, num_items_in_arg_frame
    )  # from 1 to frame size
    vacant_data_points = list(range(num_vacant_frame_data))

    for i in vacant_data_points:
        arg_frame[i] = "-"
        expected_curr_data[i] = fake_curr_data[i]

    cut = VehicleRepresentation.__new__(VehicleRepresentation)
    cut.curr_data = fake_curr_data

    # Act
    cut.update_curr_data(arg_frame)

    # Assert
    assert cut.curr_data == expected_curr_data


# get_headers tests
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
def test_VehicleRepresentation_get_current_faulting_mnemonics_returns_test_suite_call_get_status_specific_mnemonics(
    mocker,
):
    # Arrange
    expected_result = MagicMock()

    cut = VehicleRepresentation.__new__(VehicleRepresentation)
    cut.test_suite = MagicMock()

    mocker.patch.object(
        cut.test_suite, "get_status_specific_mnemonics", return_value=expected_result
    )

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

    mocker.patch.object(cut.status, "get_status", return_value=expected_result)

    # Act
    result = cut.get_status()

    # Assert
    assert cut.status.get_status.call_count == 1
    assert cut.status.get_status.call_args_list[0].args == ()
    assert result == expected_result


# get_bayesian_status tests
def test_VehicleRepresentation_get_bayesian_status_returns_status_call_to_get_bayesian_status(
    mocker,
):
    # Arrange
    expected_result = MagicMock()

    cut = VehicleRepresentation.__new__(VehicleRepresentation)
    cut.status = MagicMock()

    mocker.patch.object(cut.status, "get_bayesian_status", return_value=expected_result)

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


# get_state_information tests
def test_VehicleRepresentation_get_state_information_calls_render_reasoning_on_knowledge_synthesis_constructs(
    mocker,
):
    # Arrange
    arg_frame = MagicMock()
    arg_headers = MagicMock()
    arg_tests = MagicMock()
    fake_render_reasoning_result = MagicMock()

    fake_knowledge_synthesis_construct = MagicMock()
    fake_knowledge_synthesis_construct.component_name = "foo"

    cut = VehicleRepresentation.__new__(VehicleRepresentation)
    cut.knowledge_synthesis_constructs = [fake_knowledge_synthesis_construct]
    mocker.patch.object(
        fake_knowledge_synthesis_construct,
        "render_reasoning",
        return_value=fake_render_reasoning_result,
    )

    # Act
    result = cut.get_state_information()

    # Assert
    assert list(result.keys())[0] == "foo"
    assert list(result.values())[0] == fake_render_reasoning_result
