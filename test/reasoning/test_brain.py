""" Test Brain Functionality """
import pytest
from mock import MagicMock
import src.reasoning.brain as brain
from src.reasoning.brain import Brain


# __init__ tests
def test_Brain__init__sets_vehicle_rep_to_given_vehicle_and_learning_systems_and_mission_status_and_bayesian_status(mocker):
    # Arrange
    arg_vehicle = MagicMock()

    fake_headers = MagicMock()
    fake_learning_systems = MagicMock()
    fake_mission_status = MagicMock()
    fake_bayesian_status = MagicMock()

    mocker.patch.object(arg_vehicle, 'get_headers', return_value=fake_headers)
    mocker.patch('src.reasoning.brain.DataDrivenLearning', return_value=fake_learning_systems)
    mocker.patch.object(arg_vehicle, 'get_status', return_value=fake_mission_status)
    mocker.patch.object(arg_vehicle, 'get_bayesian_status', return_value=fake_bayesian_status)

    cut = Brain.__new__(Brain)

    # Act
    result = cut.__init__(arg_vehicle)

    # Assert
    assert cut.vehicle_rep == arg_vehicle
    assert arg_vehicle.get_headers.call_count == 1
    assert arg_vehicle.get_headers.call_args_list[0].args == ()
    assert brain.DataDrivenLearning.call_count == 1
    assert brain.DataDrivenLearning.call_args_list[0].args == (fake_headers, )
    assert cut.learning_systems == fake_learning_systems
    assert arg_vehicle.get_status.call_count == 1
    assert arg_vehicle.get_status.call_args_list[0].args == ()
    assert cut.mission_status == fake_mission_status
    assert arg_vehicle.get_bayesian_status.call_count == 1
    assert arg_vehicle.get_bayesian_status.call_args_list[0].args == ()
    assert cut.bayesian_status == fake_bayesian_status

# reason tests
def test_Brain_reason_updates_vehicle_rep_with_given_frame_and_sets_new_vehicle_mission_status_and_updates_learning_systems_with_given_frame_and_new_mission_status(mocker):
    # Arrange
    arg_frame = MagicMock()

    fake_mission_status = MagicMock()

    cut = Brain.__new__(Brain)
    cut.vehicle_rep = MagicMock()
    cut.learning_systems = MagicMock()
    
    mocker.patch.object(cut.vehicle_rep, 'update')
    mocker.patch.object(cut.vehicle_rep, 'get_status', return_value=fake_mission_status)
    mocker.patch.object(cut.learning_systems, 'update')

    # Act
    cut.reason(arg_frame)

    # Assert
    assert cut.vehicle_rep.update.call_count == 1  
    assert cut.vehicle_rep.update.call_args_list[0].args == (arg_frame, ) 
    assert cut.vehicle_rep.get_status.call_count == 1  
    assert cut.vehicle_rep.get_status.call_args_list[0].args == () 
    assert cut.learning_systems.update.call_count == 1  
    assert cut.learning_systems.update.call_args_list[0].args == (arg_frame, fake_mission_status) 
     
# diagnose tests
def test_Brain_diagnose_returns_empty_Dict():
    # Arrange
    arg_time_step = MagicMock()

    cut = Brain.__new__(Brain)
    cut.learning_systems = MagicMock()
    cut.bayesian_status = MagicMock()
    cut.vehicle_rep = MagicMock()

    # Act
    result = cut.diagnose(arg_time_step)

    # Assert
    assert type(result) == dict
    assert result == {}
