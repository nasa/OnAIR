""" Test Brain Functionality """
import pytest
from mock import MagicMock
from src.reasoning.diagnosis import Diagnosis


# __init__ tests
def test_Diagnosis__init__initializes_all_attributes_to_expected_values_when_arg_learning_system_results_is_empty_dict():

    fake_timestep = MagicMock()
    fake_learning_system_results = {}
    fake_status_confidence = MagicMock()
    fake_currently_faulting_mnemonics = MagicMock()
    fake_ground_truth = MagicMock()

    cut = Diagnosis.__new__(Diagnosis)

    # Act
    result = cut.__init__(fake_timestep,
                          fake_learning_system_results,
                          fake_status_confidence,
                          fake_currently_faulting_mnemonics,
                          fake_ground_truth)

    assert cut.time_step == fake_timestep
    assert cut.learning_system_results == fake_learning_system_results
    assert cut.status_confidence == fake_status_confidence
    assert cut.currently_faulting_mnemonics == fake_currently_faulting_mnemonics
    assert cut.ground_truth == fake_ground_truth
    assert cut.has_kalman == False
    assert cut.kalman_results == None

def test_Diagnosis__init__initializes_all_attributes_to_expected_values_when_arg_learning_system_results_does_not_contain_kalman_plugin():

    fake_timestep = MagicMock()
    fake_learning_system_results = {}
    num_learning_system_results = pytest.gen.randint(1, 10) # arbitrary, random int from 1 to 10
    for i in range(num_learning_system_results):
        fake_learning_system_results[MagicMock()] = MagicMock()
    fake_status_confidence = MagicMock()
    fake_currently_faulting_mnemonics = MagicMock()
    fake_ground_truth = MagicMock()

    cut = Diagnosis.__new__(Diagnosis)

    # Act
    result = cut.__init__(fake_timestep,
                          fake_learning_system_results,
                          fake_status_confidence,
                          fake_currently_faulting_mnemonics,
                          fake_ground_truth)

    assert cut.time_step == fake_timestep
    assert cut.learning_system_results == fake_learning_system_results
    assert cut.status_confidence == fake_status_confidence
    assert cut.currently_faulting_mnemonics == fake_currently_faulting_mnemonics
    assert cut.ground_truth == fake_ground_truth
    assert cut.has_kalman == False
    assert cut.kalman_results == None

def test_Diagnosis__init__initializes_all_attributes_to_expected_values_when_arg_learning_system_results_contains_kalman_plugin():

    fake_timestep = MagicMock()
    fake_learning_system_results = {}
    num_learning_system_results = pytest.gen.randint(0, 10) # arbitrary, random int from 0 to 10
    for i in range(num_learning_system_results):
        fake_learning_system_results[MagicMock()] = MagicMock()
    fake_kalman_results = MagicMock()
    fake_learning_system_results['kalman_plugin'] = fake_kalman_results
    fake_status_confidence = MagicMock()
    fake_currently_faulting_mnemonics = MagicMock()
    fake_ground_truth = MagicMock()

    cut = Diagnosis.__new__(Diagnosis)

    # Act
    result = cut.__init__(fake_timestep,
                          fake_learning_system_results,
                          fake_status_confidence,
                          fake_currently_faulting_mnemonics,
                          fake_ground_truth)

    assert cut.time_step == fake_timestep
    assert cut.learning_system_results == fake_learning_system_results
    assert cut.status_confidence == fake_status_confidence
    assert cut.currently_faulting_mnemonics == fake_currently_faulting_mnemonics
    assert cut.ground_truth == fake_ground_truth
    assert cut.has_kalman == True
    assert cut.kalman_results == fake_kalman_results

# perform_diagnosis tests
def test_Diagnosis_perform_diagnosis_returns_empty_Dict_when_has_kalman_is_False():
    # Arrange
    cut = Diagnosis.__new__(Diagnosis)
    cut.has_kalman = False

    # Act
    result = cut.perform_diagnosis()

    # Assert
    assert type(result) == dict
    assert result == {}

def test_Diagnosis_perform_diagnosis_returns_dict_of_str_top_and_walkdown_of_random_mnemonic_when_has_kalman_is_True(mocker):
    # Arrange
    cut = Diagnosis.__new__(Diagnosis)
    cut.has_kalman = True
    cut.kalman_results = MagicMock()

    forced_return_value = MagicMock()
    mocker.patch('src.reasoning.diagnosis.list', return_value=forced_return_value)
    mocker.patch('src.reasoning.diagnosis.random.choice', return_value=forced_return_value)
    mocker.patch.object(cut, 'walkdown', return_value=forced_return_value)

    # Act
    result = cut.perform_diagnosis()

    # Assert
    assert type(result) == dict
    assert result == {'top' : forced_return_value}

# walkdown tests
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

