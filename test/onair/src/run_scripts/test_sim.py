# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

""" Test Simulator Functionality """
import pytest
from unittest.mock import MagicMock

import onair.src.run_scripts.sim as sim
from onair.src.run_scripts.sim import Simulator

from math import ceil, floor


# constants tests
def test_Simulator_DIAGNOSIS_INTERVAL_is_expected_value():
    assert sim.DIAGNOSIS_INTERVAL == 100


# __init__ tests
def test_Simulator__init__creates_Vehicle_and_Agent(mocker):
    # Arrange
    arg_dataSource = MagicMock()
    arg_knowledge_rep_plugin_list = MagicMock()
    arg_learners_plugin_list = MagicMock()
    arg_planners_plugin_list = MagicMock()
    arg_complex_plugin_list = MagicMock()

    fake_headers = MagicMock()
    fake_tests = MagicMock()
    fake_vehicle_metadata = [fake_headers, fake_tests]
    fake_vehicle = MagicMock()
    fake_agent = MagicMock()

    cut = Simulator.__new__(Simulator)

    mocker.patch.object(
        arg_dataSource, "get_vehicle_metadata", return_value=fake_vehicle_metadata
    )
    mocker.patch(sim.__name__ + ".VehicleRepresentation", return_value=fake_vehicle)
    mocker.patch(sim.__name__ + ".Agent", return_value=fake_agent)

    # Act
    cut.__init__(
        arg_dataSource,
        arg_knowledge_rep_plugin_list,
        arg_learners_plugin_list,
        arg_planners_plugin_list,
        arg_complex_plugin_list,
    )

    # Assert
    assert cut.simData == arg_dataSource
    assert sim.VehicleRepresentation.call_count == 1
    assert sim.VehicleRepresentation.call_args_list[0].args == (
        fake_headers,
        fake_tests,
        arg_knowledge_rep_plugin_list,
    )
    assert sim.Agent.call_count == 1
    assert sim.Agent.call_args_list[0].args == (
        fake_vehicle,
        arg_learners_plugin_list,
        arg_planners_plugin_list,
        arg_complex_plugin_list,
    )
    assert cut.agent == fake_agent


# run_sim tests
def test_Simulator_run_sim_simData_never_has_more_so_loop_does_not_run_and_diagnosis_list_is_empty_but_filled_with_agent_diagnose_and_returns_last_diagnosis(
    mocker,
):
    # Arrange
    cut = Simulator.__new__(Simulator)
    cut.simData = MagicMock()
    cut.agent = MagicMock()

    fake_diagnosis = MagicMock()
    fake_time_step = 0

    mocker.patch(sim.__name__ + ".print_sim_header")
    mocker.patch(sim.__name__ + ".print_msg")
    mocker.patch.object(cut.simData, "has_more", return_value=False)
    mocker.patch.object(cut.agent, "diagnose", return_value=fake_diagnosis)

    # Act
    result = cut.run_sim()

    # Assert
    assert sim.print_sim_header.call_count == 0
    assert sim.print_msg.call_count == 0
    assert cut.simData.has_more.call_count == 1
    assert cut.simData.has_more.call_args_list[0].args == ()
    assert cut.agent.diagnose.call_count == 1
    assert cut.agent.diagnose.call_args_list[0].args == (fake_time_step,)
    assert result == fake_diagnosis


def test_Simulator_run_sim_prints_header_when_given_IO_Flag_is_equal_to_True(mocker):
    # Arrange
    cut = Simulator.__new__(Simulator)
    cut.simData = MagicMock()
    cut.agent = MagicMock()

    fake_diagnosis = MagicMock()

    mocker.patch(sim.__name__ + ".print_sim_header")
    mocker.patch(sim.__name__ + ".print_msg")
    mocker.patch.object(cut.simData, "has_more", return_value=False)
    mocker.patch.object(cut.agent, "diagnose", return_value=fake_diagnosis)

    # Act
    result = cut.run_sim(True)

    # Assert
    assert sim.print_sim_header.call_count == 1
    assert sim.print_sim_header.call_args_list[0].args == ()
    assert sim.print_msg.call_count == 0
    assert result == fake_diagnosis  # check we ran through the method correctly


def test_Simulator_run_sim_runs_until_has_more_is_false(mocker):
    # Arrange
    cut = Simulator.__new__(Simulator)
    cut.simData = MagicMock()
    cut.agent = MagicMock()

    num_fake_steps = pytest.gen.randint(1, 100)  # from 1 to 100 arbitrary for fast test
    fake_diagnosis = MagicMock()
    fake_next = MagicMock()
    fake_IO_Flag = MagicMock()
    side_effects_for_has_more = [True] * (num_fake_steps) + [False]

    mocker.patch(sim.__name__ + ".print_sim_header")
    mocker.patch(sim.__name__ + ".print_msg")
    mocker.patch.object(cut.simData, "has_more", side_effect=side_effects_for_has_more)
    mocker.patch.object(cut.simData, "get_next", return_value=fake_next)
    mocker.patch.object(cut.agent, "reason")
    mocker.patch.object(cut, "IO_check")
    mocker.patch.object(cut.agent, "mission_status", MagicMock())  # never equals 'RED'
    mocker.patch.object(cut.agent, "diagnose", return_value=fake_diagnosis)

    # Act
    result = cut.run_sim(fake_IO_Flag)

    # Assert
    assert sim.print_sim_header.call_count == 0
    assert sim.print_msg.call_count == 0
    assert cut.simData.get_next.call_count == num_fake_steps
    for i in range(num_fake_steps):
        assert cut.simData.get_next.call_args_list[i].args == ()
    assert cut.agent.reason.call_count == num_fake_steps
    for i in range(num_fake_steps):
        assert cut.agent.reason.call_args_list[i].args == (fake_next,)
    assert cut.IO_check.call_count == num_fake_steps
    for i in range(num_fake_steps):
        assert cut.IO_check.call_args_list[i].args == (
            i,
            fake_IO_Flag,
        )
    assert cut.agent.diagnose.call_count == 1
    assert cut.agent.diagnose.call_args_list[0].args == (num_fake_steps,)
    assert result == fake_diagnosis


def test_Simulator_run_sim_diagnose_always_performed_when_fault_is_on_first_time_step(
    mocker,
):
    # Arrange
    cut = Simulator.__new__(Simulator)
    cut.simData = MagicMock()
    cut.agent = MagicMock()

    fake_diagnosis = MagicMock()
    fake_next = MagicMock()
    fake_IO_Flag = MagicMock()

    mocker.patch(sim.__name__ + ".print_sim_header")
    mocker.patch(sim.__name__ + ".print_msg")
    mocker.patch.object(
        cut.simData, "has_more", side_effect=[True, False]
    )  # single loop
    mocker.patch.object(cut.simData, "get_next", return_value=fake_next)
    mocker.patch.object(cut.agent, "reason")
    mocker.patch.object(cut, "IO_check")
    mocker.patch.object(cut.agent, "mission_status", "RED")
    mocker.patch.object(cut.agent, "diagnose", return_value=fake_diagnosis)

    # Act
    result = cut.run_sim(fake_IO_Flag)

    # Assert
    assert cut.simData.get_next.call_count == 1  # verifies in loop
    assert cut.agent.reason.call_count == 1  # verifies in loop
    assert cut.IO_check.call_count == 1  # verifies in loop
    assert cut.agent.diagnose.call_count == 1
    assert cut.agent.diagnose.call_args_list[0].args == (0,)
    assert result == fake_diagnosis  # check we ran through the method correctly


def test_Simulator_run_sim_diagnose_is_not_performed_again_when_faults_are_consecutive_until_the_hundreth_step_after_last_diagnosis_and_returns_last_diagnosis(
    mocker,
):
    # Arrange
    cut = Simulator.__new__(Simulator)
    cut.simData = MagicMock()
    cut.agent = MagicMock()

    num_fake_steps = pytest.gen.randint(
        sim.DIAGNOSIS_INTERVAL, sim.DIAGNOSIS_INTERVAL * 10
    )  # from interval to (10 * interval) arbitrary
    fake_diagnoses = [MagicMock()] * (
        floor(num_fake_steps / sim.DIAGNOSIS_INTERVAL) + 1
    )  # + 1 is for last diagnosis
    fake_next = MagicMock()
    fake_IO_Flag = MagicMock()
    side_effects_for_has_more = [True] * (num_fake_steps) + [False]

    mocker.patch(sim.__name__ + ".print_sim_header")
    mocker.patch(sim.__name__ + ".print_msg")
    mocker.patch.object(cut.simData, "has_more", side_effect=side_effects_for_has_more)
    mocker.patch.object(cut.simData, "get_next", return_value=fake_next)
    mocker.patch.object(cut.agent, "reason")
    mocker.patch.object(cut, "IO_check")
    mocker.patch.object(cut.agent, "mission_status", "RED")
    mocker.patch.object(cut.agent, "diagnose", side_effect=fake_diagnoses)

    # Act
    result = cut.run_sim(fake_IO_Flag)

    # Assert
    assert cut.simData.get_next.call_count == num_fake_steps
    for i in range(num_fake_steps):
        assert cut.simData.get_next.call_args_list[i].args == ()
    assert cut.agent.reason.call_count == num_fake_steps
    for i in range(num_fake_steps):
        assert cut.agent.reason.call_args_list[i].args == (fake_next,)
    assert cut.IO_check.call_count == num_fake_steps
    for i in range(num_fake_steps):
        assert cut.IO_check.call_args_list[i].args == (
            i,
            fake_IO_Flag,
        )
    assert cut.agent.diagnose.call_count == ceil(
        num_fake_steps / sim.DIAGNOSIS_INTERVAL
    )
    for i in range(cut.agent.diagnose.call_count):
        assert cut.agent.diagnose.call_args_list[i].args == (
            i * sim.DIAGNOSIS_INTERVAL,
        )
    assert result == fake_diagnoses[-1]  # check we actually got the last diagnosis


# IO_check tests
def test_Simulator_IO_check_prints_sim_step_and_mission_status_when_given_IO_Flag_is_True(
    mocker,
):
    # Arrange
    arg_time_step = pytest.gen.randint(0, 100)  # arbitrary from 0 to 100
    arg_IO_Flag = True

    mocker.patch(sim.__name__ + ".print_sim_step")
    mocker.patch(sim.__name__ + ".print_system_status")

    cut = Simulator.__new__(Simulator)
    cut.agent = MagicMock()

    # Act
    cut.IO_check(arg_time_step, arg_IO_Flag)

    # Assert
    assert sim.print_sim_step.call_count == 1
    assert sim.print_sim_step.call_args_list[0].args == (arg_time_step + 1,)
    assert sim.print_system_status.call_count == 1
    assert sim.print_system_status.call_args_list[0].args == (
        cut.agent,
        cut.agent.vehicle_rep.curr_data,
    )


def test_Simulator_IO_check_does_nothing_when_given_IO_Flag_is_not_True(mocker):
    # Arrange
    arg_time_step = pytest.gen.randint(0, 100)  # arbitrary from 0 to 100
    arg_IO_Flag = MagicMock()

    mocker.patch(sim.__name__ + ".print_sim_step")
    mocker.patch(sim.__name__ + ".print_system_status")

    cut = Simulator.__new__(Simulator)

    # Act
    cut.IO_check(arg_time_step, arg_IO_Flag)

    # Assert
    assert sim.print_sim_step.call_count == 0
    assert sim.print_system_status.call_count == 0
