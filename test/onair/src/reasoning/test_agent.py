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
from mock import MagicMock

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
def test_Agent_reason_updates_vehicle_rep_with_given_frame_and_sets_new_vehicle_mission_status_and_updates_learning_systems_with_given_frame_and_new_mission_status(mocker):
    # Arrange
    arg_frame = MagicMock()
    arg_vehicle = MagicMock()
    arg_learners_plugin_dict = MagicMock()
    arg_planners_plugin_dict = MagicMock()
    arg_complex_plugin_dict = MagicMock()

    # Mock and patch
    fake_mission_status = MagicMock()
    fake_headers = MagicMock()
    fake_headers.__len__.return_value = 1 # Fake a header
    fake_learning_systems = MagicMock()
    fake_planning_systems = MagicMock()
    fake_complex_systems = MagicMock()
    fake_state_information = MagicMock()

    mocker.patch.object(arg_vehicle, 'get_headers', return_value=fake_headers)
    mocker.patch.object(arg_vehicle, 'update')
    mocker.patch.object(arg_vehicle, 'get_status', return_value=fake_mission_status)
    mocker.patch.object(arg_vehicle, 'get_state_information', return_value=fake_state_information)
    mocker.patch(agent.__name__ + '.LearnersInterface', return_value=fake_learning_systems)
    mocker.patch(agent.__name__ + '.PlannersInterface', return_value=fake_planning_systems)
    mocker.patch(agent.__name__ + '.ComplexReasoningInterface', return_value=fake_complex_systems)

    # Act
    cut = Agent.__new__(Agent)
    cut.__init__(arg_vehicle, arg_learners_plugin_dict, arg_planners_plugin_dict, arg_complex_plugin_dict)
    cut.reason(arg_frame)

    # Assert
    assert cut.vehicle_rep == arg_vehicle
    assert arg_vehicle.get_headers.call_count == 3
    assert cut.vehicle_rep.update.call_args_list[0].args == (arg_frame, ) 
    assert cut.vehicle_rep.get_status.call_count == 1  
    assert cut.vehicle_rep.get_status.call_args_list[0].args == () 
    assert cut.learning_systems == fake_learning_systems
    assert cut.learning_systems.update.call_count == 1  
    assert cut.learning_systems.update.call_args_list[0].args == (arg_frame, fake_state_information) 
    assert cut.planning_systems == fake_planning_systems
    assert cut.planning_systems.update.call_count == 1 
    assert cut.planning_systems.update.call_args_list[0].args == (fake_state_information, ) 
     
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
