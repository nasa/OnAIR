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
from mock import MagicMock

import onair.src.run_scripts.sim as sim
from onair.src.run_scripts.sim import Simulator

from math import ceil, floor

# __init__ tests
def test_Simulator__init__creates_Vehicle_and_AdapterDataSource_from_parsed_data_and_Agent_with_vehicle_when_SBN_Flag_resolves_to_True(mocker):
    # Arrange
    arg_simType = MagicMock()
    arg_parsedData = MagicMock()
    arg_SBN_Flag = True if (pytest.gen.randint(0,1) == 0) else MagicMock()

    class FakeDataAdapterSource:
        def __init__(self, sim_data):
            FakeDataAdapterSource.simData = self
            FakeDataAdapterSource.sim_data = sim_data
        
        def connect(self):
            if hasattr(FakeDataAdapterSource, 'connect_call_count'):
                FakeDataAdapterSource.connect_call_count += 1
            else:
                FakeDataAdapterSource.connect_call_count = 1

    fake_vehicle_metadata = [MagicMock(), MagicMock()]
    fake_vehicle = MagicMock()
    fake_sim_data = MagicMock()
    fake_sbn_adapter = MagicMock()
    fake_simData = MagicMock()
    fake_agent = MagicMock()

    cut = Simulator.__new__(Simulator)

    mocker.patch.object(arg_parsedData, 'get_vehicle_metadata', return_value=fake_vehicle_metadata)
    mocker.patch(sim.__name__ + '.vehicle', return_value=fake_vehicle)
    mocker.patch(sim.__name__ + '.importlib.import_module', return_value=fake_sbn_adapter)
    mocker.patch(sim.__name__ + '.getattr', return_value=FakeDataAdapterSource)
    mocker.patch.object(arg_parsedData, 'get_sim_data', return_value=fake_sim_data)
    mocker.patch(sim.__name__ + '.Agent', return_value=fake_agent)

    # Act
    cut.__init__(arg_simType, arg_parsedData, arg_SBN_Flag)

    # Assert
    assert cut.simulator == arg_simType
    assert sim.vehicle.call_count == 1
    assert sim.vehicle.call_args_list[0].args == (fake_vehicle_metadata[0], fake_vehicle_metadata[1], )
    assert FakeDataAdapterSource.simData == cut.simData
    assert FakeDataAdapterSource.sim_data == fake_sim_data
    assert FakeDataAdapterSource.connect_call_count == 1
    assert sim.Agent.call_count == 1
    assert sim.Agent.call_args_list[0].args == (fake_vehicle, )
    assert cut.agent == fake_agent

def test_Simulator__init__creates_Vehicle_and_DataSource_from_parsed_data_and_Agent_with_vehicle_when_SBN_Flag_resolves_to_False(mocker):
    # Arrange
    arg_simType = MagicMock()
    arg_parsedData = MagicMock()
    arg_SBN_Flag = False if (pytest.gen.randint(0,1) == 0) else None
    arg_plugin_list = MagicMock()

    fake_vehicle_metadata = [MagicMock(), MagicMock()]
    fake_vehicle = MagicMock()
    fake_sim_data = MagicMock()
    fake_simData = MagicMock()
    fake_agent = MagicMock()

    cut = Simulator.__new__(Simulator)

    mocker.patch.object(arg_parsedData, 'get_vehicle_metadata', return_value=fake_vehicle_metadata)
    mocker.patch(sim.__name__ + '.VehicleRepresentation', return_value=fake_vehicle)
    mocker.patch.object(arg_parsedData, 'get_sim_data', return_value=fake_sim_data)
    mocker.patch(sim.__name__ + '.DataSource', return_value=fake_simData)
    mocker.patch(sim.__name__ + '.Agent', return_value=fake_agent)

    # Act
    cut.__init__(arg_simType, arg_parsedData, arg_plugin_list, arg_SBN_Flag)

    # Assert
    assert cut.simulator == arg_simType
    assert sim.VehicleRepresentation.call_count == 1
    assert sim.VehicleRepresentation.call_args_list[0].args == (fake_vehicle_metadata[0], fake_vehicle_metadata[1], )
    assert sim.DataSource.call_count == 1
    assert sim.DataSource.call_args_list[0].args == (fake_sim_data, )
    assert sim.Agent.call_count == 1
    assert sim.Agent.call_args_list[0].args == (fake_vehicle, arg_plugin_list)
    assert cut.agent == fake_agent

def test_Simulator__init__creates_Vehicle_and_AdapterDataSource_from_parsed_data_and_Agent_with_vehicle_when_SBN_Flag_resolves_to_True(mocker):
    # Arrange
    arg_simType = MagicMock()
    arg_parsedData = MagicMock()
    arg_SBN_Flag = True if (pytest.gen.randint(0,1) == 0) else MagicMock()

    class FakeDataAdapterSource:
        def __init__(self, sim_data):
            FakeDataAdapterSource.simData = self
            FakeDataAdapterSource.sim_data = sim_data
        
        def connect(self):
            if hasattr(FakeDataAdapterSource, 'connect_call_count'):
                FakeDataAdapterSource.connect_call_count += 1
            else:
                FakeDataAdapterSource.connect_call_count = 1

    fake_vehicle_metadata = [MagicMock(), MagicMock()]
    fake_vehicle = MagicMock()
    fake_sim_data = MagicMock()
    fake_sbn_adapter = MagicMock()
    fake_simData = MagicMock()
    fake_agent = MagicMock()
    fake_plugin_list = MagicMock()

    cut = Simulator.__new__(Simulator)

    mocker.patch.object(arg_parsedData, 'get_vehicle_metadata', return_value=fake_vehicle_metadata)
    mocker.patch(sim.__name__ + '.VehicleRepresentation', return_value=fake_vehicle)
    mocker.patch(sim.__name__ + '.importlib.import_module', return_value=fake_sbn_adapter)
    mocker.patch(sim.__name__ + '.getattr', return_value=FakeDataAdapterSource)
    mocker.patch.object(arg_parsedData, 'get_sim_data', return_value=fake_sim_data)
    mocker.patch(sim.__name__ + '.Agent', return_value=fake_agent)

    # Act
    cut.__init__(arg_simType, arg_parsedData, fake_plugin_list, arg_SBN_Flag)

    # Assert
    assert cut.simulator == arg_simType
    assert sim.VehicleRepresentation.call_count == 1
    assert sim.VehicleRepresentation.call_args_list[0].args == (fake_vehicle_metadata[0], fake_vehicle_metadata[1], )
    assert FakeDataAdapterSource.simData == cut.simData
    assert FakeDataAdapterSource.sim_data == fake_sim_data
    assert FakeDataAdapterSource.connect_call_count == 1
    assert sim.Agent.call_count == 1
    assert sim.Agent.call_args_list[0].args == (fake_vehicle, fake_plugin_list)
    assert cut.agent == fake_agent

# run_sim tests
def test_Simulator_run_sim_simData_never_has_more_so_loop_does_not_run_and_diagnosis_list_is_empty_but_filled_with_agent_diagnose_and_returns_last_diagnosis(mocker):
    # Arrange
    cut = Simulator.__new__(Simulator)
    cut.simData = MagicMock()
    cut.agent = MagicMock()

    fake_diagnosis = MagicMock()
    fake_time_step = 0

    mocker.patch(sim.__name__ + '.print_sim_header')
    mocker.patch(sim.__name__ + '.print_msg')
    mocker.patch.object(cut.simData, 'has_more', return_value=False)
    mocker.patch.object(cut.agent, 'diagnose', return_value=fake_diagnosis)

    # Act
    result = cut.run_sim()

    # Assert
    assert sim.print_sim_header.call_count == 0
    assert sim.print_msg.call_count == 0
    assert cut.simData.has_more.call_count == 1
    assert cut.simData.has_more.call_args_list[0].args == ()
    assert cut.agent.diagnose.call_count == 1
    assert cut.agent.diagnose.call_args_list[0].args == (fake_time_step, )
    assert result == fake_diagnosis

def test_Simulator_run_sim_prints_header_when_given_IO_Flag_is_equal_to_True(mocker):
    # Arrange
    cut = Simulator.__new__(Simulator)
    cut.simData = MagicMock()
    cut.agent = MagicMock()

    fake_diagnosis = MagicMock()

    mocker.patch(sim.__name__ + '.print_sim_header')
    mocker.patch(sim.__name__ + '.print_msg')
    mocker.patch.object(cut.simData, 'has_more', return_value=False)
    mocker.patch.object(cut.agent, 'diagnose', return_value=fake_diagnosis)

    # Act
    result = cut.run_sim(True)

    # Assert
    assert sim.print_sim_header.call_count == 1
    assert sim.print_sim_header.call_args_list[0].args == ()
    assert sim.print_msg.call_count == 0
    assert result == fake_diagnosis # check we ran through the method correctly

def test_Simulator_run_sim_prints_wait_message_when_given_IO_Flag_is_the_str_strict(mocker):
    # Arrange
    cut = Simulator.__new__(Simulator)
    cut.simData = MagicMock()
    cut.agent = MagicMock()

    fake_diagnosis = MagicMock()

    mocker.patch(sim.__name__ + '.print_sim_header')
    mocker.patch(sim.__name__ + '.print_msg')
    mocker.patch.object(cut.simData, 'has_more', return_value=False)
    mocker.patch.object(cut.agent, 'diagnose', return_value=fake_diagnosis)

    # Act
    result = cut.run_sim('strict')

    # Assert
    assert sim.print_sim_header.call_count == 0
    assert sim.print_msg.call_count == 1
    assert sim.print_msg.call_args_list[0].args == ('Please wait...\n', )
    assert result == fake_diagnosis # check we ran through the method correctly

def test_Simulator_run_sim_runs_until_time_step_2050_when_simData_always_has_more(mocker):
    # Arrange
    cut = Simulator.__new__(Simulator)
    cut.simData = MagicMock()
    cut.agent = MagicMock()

    fake_diagnosis = MagicMock()
    fake_next = MagicMock()
    fake_IO_Flag = MagicMock()

    mocker.patch(sim.__name__ + '.print_sim_header')
    mocker.patch(sim.__name__ + '.print_msg')
    mocker.patch.object(cut.simData, 'has_more', return_value=True)
    mocker.patch.object(cut.simData, 'get_next', return_value=fake_next)
    mocker.patch.object(cut.agent, 'reason')
    mocker.patch.object(cut, 'IO_check')
    mocker.patch.object(cut.agent, 'mission_status', MagicMock()) # never equals 'RED'
    mocker.patch.object(cut.agent, 'diagnose', return_value=fake_diagnosis)

    # Act
    result = cut.run_sim(fake_IO_Flag)

    # Assert
    assert sim.print_sim_header.call_count == 0
    assert sim.print_msg.call_count == 0
    assert cut.simData.get_next.call_count == 2050
    for i in range(2050):
        assert cut.simData.get_next.call_args_list[i].args == ()
    assert cut.agent.reason.call_count == 2050
    for i in range(2050):
        assert cut.agent.reason.call_args_list[i].args == (fake_next, )
    assert cut.IO_check.call_count == 2050
    for i in range(2050):
        assert cut.IO_check.call_args_list[i].args == (i, fake_IO_Flag, )
    assert cut.agent.diagnose.call_count == 1
    assert cut.agent.diagnose.call_args_list[0].args == (2050, )
    assert result == fake_diagnosis

def test_Simulator_run_sim_diagnose_always_performed_when_fault_is_on_first_time_step(mocker):
    # Arrange
    cut = Simulator.__new__(Simulator)
    cut.simData = MagicMock()
    cut.agent = MagicMock()

    fake_diagnosis = MagicMock()
    fake_next = MagicMock()
    fake_IO_Flag = MagicMock()

    mocker.patch(sim.__name__ + '.print_sim_header')
    mocker.patch(sim.__name__ + '.print_msg')
    mocker.patch.object(cut.simData, 'has_more', side_effect=[True, False]) # single loop
    mocker.patch.object(cut.simData, 'get_next', return_value=fake_next)
    mocker.patch.object(cut.agent, 'reason')
    mocker.patch.object(cut, 'IO_check')
    mocker.patch.object(cut.agent, 'mission_status', 'RED')
    mocker.patch.object(cut.agent, 'diagnose', return_value=fake_diagnosis)

    # Act
    result = cut.run_sim(fake_IO_Flag)

    # Assert
    assert cut.simData.get_next.call_count == 1 # verifies in loop
    assert cut.agent.reason.call_count == 1 # verifies in loop
    assert cut.IO_check.call_count == 1 # verifies in loop
    assert cut.agent.diagnose.call_count == 1
    assert cut.agent.diagnose.call_args_list[0].args == (0, )
    assert result == fake_diagnosis # check we ran through the method correctly

def test_Simulator_run_sim_diagnose_is_not_performed_again_when_faults_are_consecutive_until_the_hundreth_step_after_last_diagnosis_and_returns_last_diagnosis(mocker):
    # Arrange
    cut = Simulator.__new__(Simulator)
    cut.simData = MagicMock()
    cut.agent = MagicMock()

    fake_diagnoses = [MagicMock()] * floor(2050/100)
    fake_diagnoses.append(MagicMock())
    fake_next = MagicMock()
    fake_IO_Flag = MagicMock()

    mocker.patch(sim.__name__ + '.print_sim_header')
    mocker.patch(sim.__name__ + '.print_msg')
    mocker.patch.object(cut.simData, 'has_more',  return_value=True) # True runs all time_steps
    mocker.patch.object(cut.simData, 'get_next', return_value=fake_next)
    mocker.patch.object(cut.agent, 'reason')
    mocker.patch.object(cut, 'IO_check')
    mocker.patch.object(cut.agent, 'mission_status', 'RED')
    mocker.patch.object(cut.agent, 'diagnose', side_effect=fake_diagnoses)

    # Act
    result = cut.run_sim(fake_IO_Flag)

    # Assert
    assert cut.simData.get_next.call_count == 2050
    for i in range(2050):
        assert cut.simData.get_next.call_args_list[i].args == ()
    assert cut.agent.reason.call_count == 2050
    for i in range(2050):
        assert cut.agent.reason.call_args_list[i].args == (fake_next, )
    assert cut.IO_check.call_count == 2050
    for i in range(2050):
        assert cut.IO_check.call_args_list[i].args == (i, fake_IO_Flag, )
    assert cut.agent.diagnose.call_count == ceil(2050/100)
    for i in range(cut.agent.diagnose.call_count):
        assert cut.agent.diagnose.call_args_list[i].args == (i * 100, )
    assert result == fake_diagnoses[-1] # check we actually got the last diagnosis

# set_benchmark_data tests
def test_Simulator_set_benchmark_data_sends_agent_supervised_learning_set_benchmark_data_given_filepath_files_and_indices(mocker):
    # Arrange
    arg_filepath = MagicMock()
    arg_files = MagicMock()
    arg_indices = MagicMock()

    cut = Simulator.__new__(Simulator)
    cut.agent = MagicMock()

    # Act
    cut.set_benchmark_data(arg_filepath, arg_files, arg_indices)

    # Assert
    assert cut.agent.supervised_learning.set_benchmark_data.call_count == 1
    assert cut.agent.supervised_learning.set_benchmark_data.call_args_list[0].args == (arg_filepath, arg_files, arg_indices, )

# IO_check tests
def test_Simulator_IO_check_prints_sim_step_and_mission_status_when_given_IO_Flag_is_True(mocker):
    # Arrange
    arg_time_step = pytest.gen.randint(0, 100) # arbitrary from 0 to 100
    arg_IO_Flag = True

    mocker.patch(sim.__name__ + '.print_sim_step')
    mocker.patch(sim.__name__ + '.print_system_status')

    cut = Simulator.__new__(Simulator)
    cut.agent = MagicMock()

    # Act
    cut.IO_check(arg_time_step, arg_IO_Flag)

    # Assert
    assert sim.print_sim_step.call_count == 1
    assert sim.print_sim_step.call_args_list[0].args == (arg_time_step + 1, )
    assert sim.print_system_status.call_count == 1
    assert sim.print_system_status.call_args_list[0].args == (cut.agent, cut.agent.vehicle_rep.curr_data, )

def test_Simulator_IO_check_does_nothing_when_given_IO_Flag_is_not_True(mocker):
    # Arrange
    arg_time_step = pytest.gen.randint(0, 100) # arbitrary from 0 to 100
    arg_IO_Flag = MagicMock()

    mocker.patch(sim.__name__ + '.print_sim_step')
    mocker.patch(sim.__name__ + '.print_system_status')

    cut = Simulator.__new__(Simulator)

    # Act
    cut.IO_check(arg_time_step, arg_IO_Flag)

    # Assert
    assert sim.print_sim_step.call_count == 0
    assert sim.print_system_status.call_count == 0
