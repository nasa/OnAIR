# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

""" Test Agent Functionality """
import pytest
from unittest.mock import MagicMock

import onair.src.reasoning.agent as agent
from onair.src.reasoning.agent import Agent

# __init__ tests
def test_Agent__init__sets_vehicle_rep_to_given_vehicle_and_learning_systems_and_mission_status_and_bayesian_status(mocker):
    # Arrange
    arg_vehicle = MagicMock()
    arg_learners_plugin_dict = MagicMock()
    arg_planners_plugin_dict = MagicMock()
    arg_complex_plugin_dict = MagicMock()

    fake_headers = MagicMock()
    fake_headers.__len__.return_value = 1 # Fake a header
    fake_learning_systems = MagicMock()
    fake_planning_systems = MagicMock()
    fake_complex_systems = MagicMock()
    fake_mission_status = MagicMock()
    fake_bayesian_status = MagicMock()

    mocker.patch.object(arg_vehicle, 'get_headers', return_value=fake_headers)
    mocker.patch.object(arg_vehicle, 'get_status', return_value=fake_mission_status)
    mocker.patch.object(arg_vehicle, 'get_bayesian_status', return_value=fake_bayesian_status)
    mocker.patch(agent.__name__ + '.LearnersInterface', return_value=fake_learning_systems)
    mocker.patch(agent.__name__ + '.PlannersInterface', return_value=fake_planning_systems)
    mocker.patch(agent.__name__ + '.ComplexReasoningInterface', return_value=fake_complex_systems)

    cut = Agent.__new__(Agent)

    # Act
    cut.__init__(arg_vehicle, arg_learners_plugin_dict, arg_planners_plugin_dict, arg_complex_plugin_dict)

    # Assert
    assert cut.vehicle_rep == arg_vehicle
    assert arg_vehicle.get_headers.call_count == 3
    assert arg_vehicle.get_headers.call_args_list[0].args == ()
    assert agent.LearnersInterface.call_count == 1
    assert agent.LearnersInterface.call_args_list[0].args == (fake_headers, arg_learners_plugin_dict)
    assert cut.learning_systems == fake_learning_systems
    assert agent.PlannersInterface.call_count == 1
    assert agent.PlannersInterface.call_args_list[0].args == (fake_headers, arg_planners_plugin_dict)
    assert cut.planning_systems == fake_planning_systems
    assert agent.ComplexReasoningInterface.call_count == 1
    assert agent.ComplexReasoningInterface.call_args_list[0].args == (fake_headers, arg_complex_plugin_dict)
    assert cut.complex_reasoning_systems == fake_complex_systems
    assert arg_vehicle.get_status.call_count == 1
    assert arg_vehicle.get_status.call_args_list[0].args == ()
    assert cut.mission_status == fake_mission_status
    assert arg_vehicle.get_bayesian_status.call_count == 1
    assert arg_vehicle.get_bayesian_status.call_args_list[0].args == ()
    assert cut.bayesian_status == fake_bayesian_status

# reason tests
def test_Agent_reason_updates_vehicle_rep_with_given_frame_learners_with_frame_and_aggregated_high_level_data_planners_with_aggreagated_high_level_data_returning_complex_reasonings_update_and_render_reasoning(mocker):
    # Arrange
    arg_frame = MagicMock()
    fake_vehicle_rep = MagicMock()
    fake_vehicle_rep.curr_data = MagicMock()

    # Mock and patch
    fake_status = MagicMock()
    fake_vehicle_rep_state = MagicMock()
    fake_learning_systems = MagicMock()
    fake_planning_systems = MagicMock()
    fake_complex_reasoning_systems = MagicMock()
    fake_learning_systems_reasoning = MagicMock()
    fake_planning_systems_reasoning = MagicMock()
    expected_aggregate_to_learners = {'vehicle_rep': fake_vehicle_rep_state}
    expected_aggregate_to_planners = {'vehicle_rep': fake_vehicle_rep_state,
                                      'learning_systems':fake_learning_systems_reasoning}
    expected_aggregate_to_complex = {'vehicle_rep': fake_vehicle_rep_state,
                                     'learning_systems':fake_learning_systems_reasoning,
                                     'planning_systems':fake_planning_systems_reasoning}
    expected_result = MagicMock()

    cut = Agent.__new__(Agent)
    cut.vehicle_rep = fake_vehicle_rep
    cut.learning_systems = fake_learning_systems
    cut.planning_systems = fake_planning_systems
    cut.complex_reasoning_systems = fake_complex_reasoning_systems

    mock_manager = mocker.MagicMock()

    mock_manager.attach_mock(mocker.patch.object(fake_vehicle_rep, 'update'), 'cut.vehicle_rep.update')
    mock_manager.attach_mock(mocker.patch.object(fake_vehicle_rep, 'get_state_information', return_value=fake_vehicle_rep_state), 'cut.vehicle_rep.get_state_information')
    mock_manager.attach_mock(mocker.patch.object(fake_learning_systems, 'update'), 'cut.learning_systems.update')
    mock_manager.attach_mock(mocker.patch.object(fake_learning_systems, 'render_reasoning', return_value=fake_learning_systems_reasoning), 'cut.learning_systems.render_reasoning')
    mock_manager.attach_mock(mocker.patch.object(fake_planning_systems, 'update'), 'cut.planning_systems.update')
    mock_manager.attach_mock(mocker.patch.object(fake_planning_systems, 'render_reasoning', return_value=fake_planning_systems_reasoning), 'cut.planning_systems.render_reasoning')
    mock_manager.attach_mock(mocker.patch.object(fake_complex_reasoning_systems, 'update_and_render_reasoning', return_value=expected_result), 'cut.complex_reasoning_systems.update_and_render_reasoning')


    # mocker.patch.object(fake_learning_systems, 'render_reasoning', return_value=fake_learning_systems_reasoning)
    # mocker.patch.object(fake_planning_systems, 'render_reasoning', return_value=fake_planning_systems_reasoning)

    # Act
    result = cut.reason(arg_frame)

    # Assert
    result = expected_result
    #TODO: using expected_aggregate_to_complex is incorrect, appears to maybe be an issue with MagicMock somehow
    # problem is its always the same object, that gets updated during the function, unfortunately it only saves the object
    # not a "snapshot" of what the object was at the time, so each recorded call thinks it got the object which it did, but the state is wrong
    # side_effect could be used to save the true values, but research better options
    mock_manager.assert_has_calls([
        mocker.call.cut.vehicle_rep.update(arg_frame),
        mocker.call.cut.vehicle_rep.get_state_information(),
        mocker.call.cut.learning_systems.update(fake_vehicle_rep.curr_data, expected_aggregate_to_complex),
        mocker.call.cut.learning_systems.render_reasoning(),
        mocker.call.cut.planning_systems.update(expected_aggregate_to_complex),
        mocker.call.cut.planning_systems.render_reasoning(),
        mocker.call.cut.complex_reasoning_systems.update_and_render_reasoning(expected_aggregate_to_complex),
    ], any_order=False)


# diagnose tests
def test_Agent_diagnose_returns_empty_Dict():
    # Arrange
    arg_time_step = MagicMock()

    cut = Agent.__new__(Agent)
    cut.learning_systems = MagicMock()
    cut.planning_systems = MagicMock()
    cut.bayesian_status = MagicMock()
    cut.vehicle_rep = MagicMock()

    # Act
    result = cut.diagnose(arg_time_step)

    # Assert
    assert type(result) == dict
    assert result == {}
