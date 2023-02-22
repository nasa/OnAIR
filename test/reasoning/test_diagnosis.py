""" Test Brain Functionality """
import pytest
from mock import MagicMock
from src.reasoning.diagnosis import Diagnosis


# __init__ tests
def test_Diagnose__init__sets_spacecraft_rep_to_given_spacecraft_and_learning_systems_and_mission_status_and_bayesian_status(mocker):

    fake_timestep = MagicMock()
    fake_learning_system_results = MagicMock()
    fake_status_confidence = MagicMock()
    fake_currently_faulting_mnemonics = MagicMock()
    fake_ground_truth = MagicMock()

    cut = Diagnosis.__new__(Diagnosis)

    # Act
    result = cut.__init__(fake_timestep,
                          fake_learning_system_results,
                          fake_status_confidence,
                          fake_currently_faulting_mnemonics,
                          fake_ground_truth
                          )

    assert cut.time_step == fake_timestep
    assert cut.learning_system_results == fake_learning_system_results
    assert cut.status_confidence == fake_status_confidence
    assert cut.currently_faulting_mnemonics == fake_currently_faulting_mnemonics
    assert cut.ground_truth == fake_ground_truth
     
# diagnose tests
def test_Diagnose_returns_empty_Dict():
    # Arrange
    arg_time_step = MagicMock()

    cut = Diagnosis.__new__(Diagnosis)
    cut.learning_systems = MagicMock()
    cut.bayesian_status = MagicMock()
    cut.spacecraft_rep = MagicMock()
    cut.has_kalman = False
    cut.kalman_results = MagicMock()

    # Act
    result = cut.perform_diagnosis()

    # Assert
    assert type(result) == dict
    assert result == {}

def test_Diagnosis_walkdown_returns_NODIAGNOSIS():
    # Arrange
    arg_time_step = MagicMock()

    cut = Diagnosis.__new__(Diagnosis)
    cut.time_step = MagicMock()
    cut.status_confidence = MagicMock()
    cut.learning_system_results = MagicMock()
    cut.currently_faulting_mnemonics = MagicMock()
    cut.ground_truth = MagicMock()
    cut.has_kalman = False
    cut.kalman_results = MagicMock()

    # Act
    fake_mnemonic = "fake_mnemonic"
    result = cut.walkdown(fake_mnemonic)
    assert result == Diagnosis.NO_DIAGNOSIS

