""" Test Simulator Functionality """ 
import pytest
from mock import MagicMock
import src.run_scripts.sim as sim
from src.run_scripts.sim import Simulator
from math import ceil, floor

# __init__ tests
def test__init__creates_Spacecraft_and_AdapterDataSource_from_parsed_data_and_Brain_with_spaceCraft_when_SBN_Flag_resolves_to_True(mocker):
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

    fake_spacecraft_metadata = [MagicMock(), MagicMock()]
    fake_spaceCraft = MagicMock()
    fake_sim_data = MagicMock()
    fake_sbn_adapter = MagicMock()
    fake_simData = MagicMock()
    fake_brain = MagicMock()

    cut = Simulator.__new__(Simulator)

    mocker.patch.object(arg_parsedData, 'get_spacecraft_metadata', return_value=fake_spacecraft_metadata)
    mocker.patch('src.run_scripts.sim.Spacecraft', return_value=fake_spaceCraft)
    mocker.patch('src.run_scripts.sim.importlib.import_module', return_value=fake_sbn_adapter)
    mocker.patch('src.run_scripts.sim.getattr', return_value=FakeDataAdapterSource)
    mocker.patch.object(arg_parsedData, 'get_sim_data', return_value=fake_sim_data)
    mocker.patch('src.run_scripts.sim.Brain', return_value=fake_brain)

    # Act
    cut.__init__(arg_simType, arg_parsedData, arg_SBN_Flag)

    # Assert
    assert cut.simulator == arg_simType
    assert sim.Spacecraft.call_count == 1
    assert sim.Spacecraft.call_args_list[0].args == (fake_spacecraft_metadata[0], fake_spacecraft_metadata[1], )
    assert FakeDataAdapterSource.simData == cut.simData
    assert FakeDataAdapterSource.sim_data == fake_sim_data
    assert FakeDataAdapterSource.connect_call_count == 1
    assert sim.Brain.call_count == 1
    assert sim.Brain.call_args_list[0].args == (fake_spaceCraft, )
    assert cut.brain == fake_brain

def test__init__creates_Spacecraft_and_DataSource_from_parsed_data_and_Brain_with_spaceCraft_when_SBN_Flag_resolves_to_False(mocker):
    # Arrange
    arg_simType = MagicMock()
    arg_parsedData = MagicMock()
    arg_SBN_Flag = False if (pytest.gen.randint(0,1) == 0) else None

    fake_spacecraft_metadata = [MagicMock(), MagicMock()]
    fake_spaceCraft = MagicMock()
    fake_sim_data = MagicMock()
    fake_simData = MagicMock()
    fake_brain = MagicMock()

    cut = Simulator.__new__(Simulator)

    mocker.patch.object(arg_parsedData, 'get_spacecraft_metadata', return_value=fake_spacecraft_metadata)
    mocker.patch('src.run_scripts.sim.Spacecraft', return_value=fake_spaceCraft)
    mocker.patch.object(arg_parsedData, 'get_sim_data', return_value=fake_sim_data)
    mocker.patch('src.run_scripts.sim.DataSource', return_value=fake_simData)
    mocker.patch('src.run_scripts.sim.Brain', return_value=fake_brain)

    # Act
    cut.__init__(arg_simType, arg_parsedData, arg_SBN_Flag)

    # Assert
    assert cut.simulator == arg_simType
    assert sim.Spacecraft.call_count == 1
    assert sim.Spacecraft.call_args_list[0].args == (fake_spacecraft_metadata[0], fake_spacecraft_metadata[1], )
    assert sim.DataSource.call_count == 1
    assert sim.DataSource.call_args_list[0].args == (fake_sim_data, )
    assert sim.Brain.call_count == 1
    assert sim.Brain.call_args_list[0].args == (fake_spaceCraft, )
    assert cut.brain == fake_brain

def test__init__creates_Spacecraft_and_AdapterDataSource_from_parsed_data_and_Brain_with_spaceCraft_when_SBN_Flag_resolves_to_True(mocker):
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

    fake_spacecraft_metadata = [MagicMock(), MagicMock()]
    fake_spaceCraft = MagicMock()
    fake_sim_data = MagicMock()
    fake_sbn_adapter = MagicMock()
    fake_simData = MagicMock()
    fake_brain = MagicMock()

    cut = Simulator.__new__(Simulator)

    mocker.patch.object(arg_parsedData, 'get_spacecraft_metadata', return_value=fake_spacecraft_metadata)
    mocker.patch('src.run_scripts.sim.Spacecraft', return_value=fake_spaceCraft)
    mocker.patch('src.run_scripts.sim.importlib.import_module', return_value=fake_sbn_adapter)
    mocker.patch('src.run_scripts.sim.getattr', return_value=FakeDataAdapterSource)
    mocker.patch.object(arg_parsedData, 'get_sim_data', return_value=fake_sim_data)
    mocker.patch('src.run_scripts.sim.Brain', return_value=fake_brain)

    # Act
    cut.__init__(arg_simType, arg_parsedData, arg_SBN_Flag)

    # Assert
    assert cut.simulator == arg_simType
    assert sim.Spacecraft.call_count == 1
    assert sim.Spacecraft.call_args_list[0].args == (fake_spacecraft_metadata[0], fake_spacecraft_metadata[1], )
    assert FakeDataAdapterSource.simData == cut.simData
    assert FakeDataAdapterSource.sim_data == fake_sim_data
    assert FakeDataAdapterSource.connect_call_count == 1
    assert sim.Brain.call_count == 1
    assert sim.Brain.call_args_list[0].args == (fake_spaceCraft, )
    assert cut.brain == fake_brain

# run_sim tests
def test_run_sim_simData_never_has_more_so_loop_does_not_run_and_diagnosis_list_is_empty_but_filled_with_brain_diagnose_and_returns_last_diagnosis(mocker):
    # Arrange
    cut = Simulator.__new__(Simulator)
    cut.simData = MagicMock()
    cut.brain = MagicMock()

    fake_diagnosis = MagicMock()
    fake_time_step = 0

    mocker.patch('src.run_scripts.sim.print_sim_header')
    mocker.patch('src.run_scripts.sim.print_msg')
    mocker.patch.object(cut.simData, 'has_more', return_value=False)
    mocker.patch.object(cut.brain, 'diagnose', return_value=fake_diagnosis)

    # Act
    result = cut.run_sim()

    # Assert
    assert sim.print_sim_header.call_count == 0
    assert sim.print_msg.call_count == 0
    assert cut.simData.has_more.call_count == 1
    assert cut.simData.has_more.call_args_list[0].args == ()
    assert cut.brain.diagnose.call_count == 1
    assert cut.brain.diagnose.call_args_list[0].args == (fake_time_step, )
    assert result == fake_diagnosis

def test_run_sim_prints_header_when_given_IO_Flag_is_equal_to_True(mocker):
    # Arrange
    cut = Simulator.__new__(Simulator)
    cut.simData = MagicMock()
    cut.brain = MagicMock()

    fake_diagnosis = MagicMock()

    mocker.patch('src.run_scripts.sim.print_sim_header')
    mocker.patch('src.run_scripts.sim.print_msg')
    mocker.patch.object(cut.simData, 'has_more', return_value=False)
    mocker.patch.object(cut.brain, 'diagnose', return_value=fake_diagnosis)

    # Act
    result = cut.run_sim(True)

    # Assert
    assert sim.print_sim_header.call_count == 1
    assert sim.print_sim_header.call_args_list[0].args == ()
    assert sim.print_msg.call_count == 0
    assert result == fake_diagnosis # check we ran through the method correctly

def test_run_sim_prints_wait_message_when_given_IO_Flag_is_the_str_strict(mocker):
    # Arrange
    cut = Simulator.__new__(Simulator)
    cut.simData = MagicMock()
    cut.brain = MagicMock()

    fake_diagnosis = MagicMock()

    mocker.patch('src.run_scripts.sim.print_sim_header')
    mocker.patch('src.run_scripts.sim.print_msg')
    mocker.patch.object(cut.simData, 'has_more', return_value=False)
    mocker.patch.object(cut.brain, 'diagnose', return_value=fake_diagnosis)

    # Act
    result = cut.run_sim('strict')

    # Assert
    assert sim.print_sim_header.call_count == 0
    assert sim.print_msg.call_count == 1
    assert sim.print_msg.call_args_list[0].args == ('Please wait...\n', )
    assert result == fake_diagnosis # check we ran through the method correctly

def test_run_sim_runs_until_time_step_2050_when_simData_always_has_more(mocker):
    # Arrange
    cut = Simulator.__new__(Simulator)
    cut.simData = MagicMock()
    cut.brain = MagicMock()

    fake_diagnosis = MagicMock()
    fake_next = MagicMock()
    fake_IO_Flag = MagicMock()

    mocker.patch('src.run_scripts.sim.print_sim_header')
    mocker.patch('src.run_scripts.sim.print_msg')
    mocker.patch.object(cut.simData, 'has_more', return_value=True)
    mocker.patch.object(cut.simData, 'get_next', return_value=fake_next)
    mocker.patch.object(cut.brain, 'reason')
    mocker.patch.object(cut, 'IO_check')
    mocker.patch.object(cut.brain, 'mission_status', MagicMock()) # never equals 'RED'
    mocker.patch.object(cut.brain, 'diagnose', return_value=fake_diagnosis)

    # Act
    result = cut.run_sim(fake_IO_Flag)

    # Assert
    assert sim.print_sim_header.call_count == 0
    assert sim.print_msg.call_count == 0
    assert cut.simData.get_next.call_count == 2050
    for i in range(2050):
        assert cut.simData.get_next.call_args_list[i].args == ()
    assert cut.brain.reason.call_count == 2050
    for i in range(2050):
        assert cut.brain.reason.call_args_list[i].args == (fake_next, )
    assert cut.IO_check.call_count == 2050
    for i in range(2050):
        assert cut.IO_check.call_args_list[i].args == (i, fake_IO_Flag, )
    assert cut.brain.diagnose.call_count == 1
    assert cut.brain.diagnose.call_args_list[0].args == (2050, )
    assert result == fake_diagnosis

def test_run_sim_diagnose_always_performed_when_fault_is_on_first_time_step(mocker):
    # Arrange
    cut = Simulator.__new__(Simulator)
    cut.simData = MagicMock()
    cut.brain = MagicMock()

    fake_diagnosis = MagicMock()
    fake_next = MagicMock()
    fake_IO_Flag = MagicMock()

    mocker.patch('src.run_scripts.sim.print_sim_header')
    mocker.patch('src.run_scripts.sim.print_msg')
    mocker.patch.object(cut.simData, 'has_more', side_effect=[True, False]) # single loop
    mocker.patch.object(cut.simData, 'get_next', return_value=fake_next)
    mocker.patch.object(cut.brain, 'reason')
    mocker.patch.object(cut, 'IO_check')
    mocker.patch.object(cut.brain, 'mission_status', 'RED')
    mocker.patch.object(cut.brain, 'diagnose', return_value=fake_diagnosis)

    # Act
    result = cut.run_sim(fake_IO_Flag)

    # Assert
    assert cut.simData.get_next.call_count == 1 # verifies in loop
    assert cut.brain.reason.call_count == 1 # verifies in loop
    assert cut.IO_check.call_count == 1 # verifies in loop
    assert cut.brain.diagnose.call_count == 1
    assert cut.brain.diagnose.call_args_list[0].args == (0, )
    assert result == fake_diagnosis # check we ran through the method correctly

def test_run_sim_diagnose_is_not_performed_again_when_faults_are_consecutive_until_the_hundreth_step_after_last_diagnosis_and_returns_last_diagnosis(mocker):
    # Arrange
    cut = Simulator.__new__(Simulator)
    cut.simData = MagicMock()
    cut.brain = MagicMock()

    fake_diagnoses = [MagicMock()] * floor(2050/100)
    fake_diagnoses.append(MagicMock())
    fake_next = MagicMock()
    fake_IO_Flag = MagicMock()

    mocker.patch('src.run_scripts.sim.print_sim_header')
    mocker.patch('src.run_scripts.sim.print_msg')
    mocker.patch.object(cut.simData, 'has_more',  return_value=True) # True runs all time_steps
    mocker.patch.object(cut.simData, 'get_next', return_value=fake_next)
    mocker.patch.object(cut.brain, 'reason')
    mocker.patch.object(cut, 'IO_check')
    mocker.patch.object(cut.brain, 'mission_status', 'RED')
    mocker.patch.object(cut.brain, 'diagnose', side_effect=fake_diagnoses)

    # Act
    result = cut.run_sim(fake_IO_Flag)

    # Assert
    assert cut.simData.get_next.call_count == 2050
    for i in range(2050):
        assert cut.simData.get_next.call_args_list[i].args == ()
    assert cut.brain.reason.call_count == 2050
    for i in range(2050):
        assert cut.brain.reason.call_args_list[i].args == (fake_next, )
    assert cut.IO_check.call_count == 2050
    for i in range(2050):
        assert cut.IO_check.call_args_list[i].args == (i, fake_IO_Flag, )
    assert cut.brain.diagnose.call_count == ceil(2050/100)
    for i in range(cut.brain.diagnose.call_count):
        assert cut.brain.diagnose.call_args_list[i].args == (i * 100, )
    assert result == fake_diagnoses[-1] # check we actually got the last diagnosis

# set_benchmark_data tests
def test_set_benchmark_data_sends_brain_supervised_learning_set_benchmark_data_given_filepath_files_and_indices(mocker):
    # Arrange
    arg_filepath = MagicMock()
    arg_files = MagicMock()
    arg_indices = MagicMock()

    cut = Simulator.__new__(Simulator)
    cut.brain = MagicMock()

    # Act
    cut.set_benchmark_data(arg_filepath, arg_files, arg_indices)

    # Assert
    assert cut.brain.supervised_learning.set_benchmark_data.call_count == 1
    assert cut.brain.supervised_learning.set_benchmark_data.call_args_list[0].args == (arg_filepath, arg_files, arg_indices, )

# IO_check tests
def test_IO_check_prints_sim_step_and_mission_status_when_given_IO_Flag_is_True(mocker):
    # Arrange
    arg_time_step = pytest.gen.randint(0, 100) # arbitrary from 0 to 100
    arg_IO_Flag = True

    mocker.patch('src.run_scripts.sim.print_sim_step')
    mocker.patch('src.run_scripts.sim.print_mission_status')

    cut = Simulator.__new__(Simulator)
    cut.brain = MagicMock()

    # Act
    cut.IO_check(arg_time_step, arg_IO_Flag)

    # Assert
    assert sim.print_sim_step.call_count == 1
    assert sim.print_sim_step.call_args_list[0].args == (arg_time_step + 1, )
    assert sim.print_mission_status.call_count == 1
    assert sim.print_mission_status.call_args_list[0].args == (cut.brain, cut.brain.spacecraft_rep.curr_data, )

def test_IO_check_does_nothing_when_given_IO_Flag_is_not_True(mocker):
    # Arrange
    arg_time_step = pytest.gen.randint(0, 100) # arbitrary from 0 to 100
    arg_IO_Flag = MagicMock()

    mocker.patch('src.run_scripts.sim.print_sim_step')
    mocker.patch('src.run_scripts.sim.print_mission_status')

    cut = Simulator.__new__(Simulator)

    # Act
    cut.IO_check(arg_time_step, arg_IO_Flag)

    # Assert
    assert sim.print_sim_step.call_count == 0
    assert sim.print_mission_status.call_count == 0
