# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

""" Test Diagnosis Functionality """
import pytest
from unittest.mock import MagicMock

import onair.src.reasoning.diagnosis as diagnosis
from onair.src.reasoning.diagnosis import Diagnosis


# __init__ tests
def test_Diagnosis__init__initializes_all_attributes_to_expected_values_when_arg_learning_system_results_is_empty_dict():
    # Assert
    fake_timestep = MagicMock()
    fake_learning_system_results = {}
    fake_status_confidence = MagicMock()
    fake_currently_faulting_mnemonics = MagicMock()
    fake_ground_truth = MagicMock()

    cut = Diagnosis.__new__(Diagnosis)

    # Act
    result = cut.__init__(
        fake_timestep,
        fake_learning_system_results,
        fake_status_confidence,
        fake_currently_faulting_mnemonics,
        fake_ground_truth,
    )

    # Assert
    assert cut.time_step == fake_timestep
    assert cut.learning_system_results == fake_learning_system_results
    assert cut.status_confidence == fake_status_confidence
    assert cut.currently_faulting_mnemonics == fake_currently_faulting_mnemonics
    assert cut.ground_truth == fake_ground_truth
    assert cut.has_kalman == False
    assert cut.kalman_results == None


def test_Diagnosis__init__initializes_all_attributes_to_expected_values_when_arg_learning_system_results_is_non_empty_and_does_not_contain_kalman_plugin():
    # Arrange
    fake_timestep = MagicMock()
    fake_learning_system_results = {}
    num_learning_system_results = pytest.gen.randint(
        1, 10
    )  # arbitrary, random int from 1 to 10
    for i in range(num_learning_system_results):
        fake_learning_system_results[MagicMock()] = MagicMock()
    fake_status_confidence = MagicMock()
    fake_currently_faulting_mnemonics = MagicMock()
    fake_ground_truth = MagicMock()

    cut = Diagnosis.__new__(Diagnosis)

    # Act
    result = cut.__init__(
        fake_timestep,
        fake_learning_system_results,
        fake_status_confidence,
        fake_currently_faulting_mnemonics,
        fake_ground_truth,
    )

    # Assert
    assert cut.time_step == fake_timestep
    assert cut.learning_system_results == fake_learning_system_results
    assert cut.status_confidence == fake_status_confidence
    assert cut.currently_faulting_mnemonics == fake_currently_faulting_mnemonics
    assert cut.ground_truth == fake_ground_truth
    assert cut.has_kalman == False
    assert cut.kalman_results == None


def test_Diagnosis__init__initializes_all_attributes_to_expected_values_when_arg_learning_system_results_is_non_empty_and_contains_kalman_plugin():
    # Arrange
    fake_timestep = MagicMock()
    fake_learning_system_results = {}
    num_learning_system_results = pytest.gen.randint(
        0, 10
    )  # arbitrary, random int from 0 to 10
    for i in range(num_learning_system_results):
        fake_learning_system_results[MagicMock()] = MagicMock()
    fake_kalman_results = MagicMock()
    fake_learning_system_results["kalman"] = fake_kalman_results
    fake_status_confidence = MagicMock()
    fake_currently_faulting_mnemonics = MagicMock()
    fake_ground_truth = MagicMock()

    cut = Diagnosis.__new__(Diagnosis)

    # Act
    result = cut.__init__(
        fake_timestep,
        fake_learning_system_results,
        fake_status_confidence,
        fake_currently_faulting_mnemonics,
        fake_ground_truth,
    )

    # Assert
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


def test_Diagnosis_perform_diagnosis_returns_dict_of_str_top_and_walkdown_of_random_mnemonic_when_has_kalman_is_True(
    mocker,
):
    # Arrange
    fake_kalman_results = MagicMock()

    cut = Diagnosis.__new__(Diagnosis)
    cut.has_kalman = True
    cut.kalman_results = fake_kalman_results

    forced_list_return_value = MagicMock()
    forced_random_choice_return_value = MagicMock()
    forced_walkdown_return_value = MagicMock()
    mocker.patch(diagnosis.__name__ + ".list", return_value=forced_list_return_value)
    mocker.patch(
        diagnosis.__name__ + ".random.choice",
        return_value=forced_random_choice_return_value,
    )
    mocker.patch.object(cut, "walkdown", return_value=forced_walkdown_return_value)

    # Act
    result = cut.perform_diagnosis()

    # Assert
    assert type(result) == dict
    assert result == {"top": forced_walkdown_return_value}
    assert diagnosis.list.call_count == 1
    assert diagnosis.list.call_args_list[0].args == (fake_kalman_results[0],)
    assert diagnosis.random.choice.call_count == 1
    assert diagnosis.random.choice.call_args_list[0].args == (forced_list_return_value,)
    assert cut.walkdown.call_count == 1
    assert cut.walkdown.call_args_list[0].args == (forced_random_choice_return_value,)


# walkdown tests
def test_Diagnosis_walkdown_returns_expected_value_and_does_not_call_copy_deepcopy_function_when_used_mnemonics_is_not_empty_and_mnemonic_name_is_not_blank_and_has_kalman_is_True_and_kalman_results_does_not_contain_mnemonic_name(
    mocker,
):
    # Arrange
    arg_mnemonic_name = str(MagicMock())
    num_used_mnemonics = pytest.gen.randint(1, 10)  # arbitrary, random int from 1 to 10
    arg_used_mnemonics = [MagicMock()] * num_used_mnemonics

    fake_kalman_results = [MagicMock()] * pytest.gen.randint(
        1, 10
    )  # arbitrary, random int from 1 to 10
    len_fake_list = pytest.gen.randint(1, 10)  # arbitrary, random int from 1 to 10
    fake_kalman_results[0] = [MagicMock()] * len_fake_list

    expected_result = fake_kalman_results[0][0]

    cut = Diagnosis.__new__(Diagnosis)
    cut.has_kalman = True
    cut.kalman_results = fake_kalman_results

    mocker.patch(diagnosis.__name__ + ".copy.deepcopy")

    # Act
    result = cut.walkdown(arg_mnemonic_name, arg_used_mnemonics)

    # Assert
    assert result == expected_result
    assert diagnosis.copy.deepcopy.call_count == 0


def test_Diagnosis_walkdown_returns_expected_value_and_calls_copy_deepcopy_function_when_used_mnemonics_is_empty_and_mnemonic_name_is_not_blank_and_has_kalman_is_True_and_kalman_results_does_not_contain_mnemonic_name(
    mocker,
):
    # Arrange
    arg_mnemonic_name = str(MagicMock())
    arg_used_mnemonics = []

    fake_kalman_results = [MagicMock()] * pytest.gen.randint(
        1, 10
    )  # arbitrary, random int from 1 to 10
    len_fake_list = pytest.gen.randint(1, 10)  # arbitrary, random int from 1 to 10
    fake_kalman_results[0] = [MagicMock()] * len_fake_list
    fake_currently_faulting_mnemonics = MagicMock()

    expected_result = fake_kalman_results[0][0]

    cut = Diagnosis.__new__(Diagnosis)
    cut.currently_faulting_mnemonics = fake_currently_faulting_mnemonics
    cut.has_kalman = True
    cut.kalman_results = fake_kalman_results

    mocker.patch(diagnosis.__name__ + ".copy.deepcopy")

    # Act
    result = cut.walkdown(arg_mnemonic_name, arg_used_mnemonics)

    # Assert
    assert result == expected_result
    assert diagnosis.copy.deepcopy.call_count == 1
    assert diagnosis.copy.deepcopy.call_args_list[0].args == (
        fake_currently_faulting_mnemonics,
    )


def test_Diagnosis_walkdown_returns_NO_DIAGNOSIS_when_mnemonic_name_is_not_blank_and_has_kalman_is_True_and_kalman_results_contains_mnemonic_name(
    mocker,
):
    # Arrange
    arg_mnemonic_name = str(MagicMock())
    num_used_mnemonics = pytest.gen.randint(1, 10)  # arbitrary, random int from 1 to 10
    arg_used_mnemonics = [MagicMock()] * num_used_mnemonics

    fake_kalman_results = [MagicMock()] * pytest.gen.randint(
        1, 10
    )  # arbitrary, random int from 1 to 10
    len_fake_list = pytest.gen.randint(1, 10)  # arbitrary, random int from 1 to 10
    fake_kalman_results[0] = [MagicMock()] * len_fake_list

    rand_name_index = pytest.gen.randint(
        0, len_fake_list - 1
    )  # random index in fake_kalman_results[0]
    fake_kalman_results[0][rand_name_index] = arg_mnemonic_name

    cut = Diagnosis.__new__(Diagnosis)
    cut.has_kalman = True
    cut.kalman_results = fake_kalman_results

    mocker.patch(diagnosis.__name__ + ".copy.deepcopy")

    # Act
    result = cut.walkdown(arg_mnemonic_name, arg_used_mnemonics)

    # Assert
    assert result == Diagnosis.NO_DIAGNOSIS
    assert diagnosis.copy.deepcopy.call_count == 0


def test_Diagnosis_walkdown_returns_NO_DIAGNOSIS_when_mnemonic_name_is_not_blank_and_has_kalman_is_False_and_kalman_results_does_not_contain_mnemonic_name(
    mocker,
):
    # Arrange
    arg_mnemonic_name = str(MagicMock())
    num_used_mnemonics = pytest.gen.randint(1, 10)  # arbitrary, random int from 1 to 10
    arg_used_mnemonics = [MagicMock()] * num_used_mnemonics

    fake_kalman_results = [MagicMock()] * pytest.gen.randint(
        1, 10
    )  # arbitrary, random int from 1 to 10
    len_fake_list = pytest.gen.randint(1, 10)  # arbitrary, random int from 1 to 10
    fake_kalman_results[0] = [MagicMock()] * len_fake_list

    cut = Diagnosis.__new__(Diagnosis)
    cut.has_kalman = False
    cut.kalman_results = fake_kalman_results

    mocker.patch(diagnosis.__name__ + ".copy.deepcopy")

    # Act
    result = cut.walkdown(arg_mnemonic_name, arg_used_mnemonics)

    # Assert
    assert result == Diagnosis.NO_DIAGNOSIS
    assert diagnosis.copy.deepcopy.call_count == 0


def test_Diagnosis_walkdown_returns_NO_DIAGNOSIS_when_mnemonic_name_is_blank_and_has_kalman_is_True_and_kalman_results_does_not_contain_mnemonic_name(
    mocker,
):
    # Arrange
    arg_mnemonic_name = ""
    num_used_mnemonics = pytest.gen.randint(1, 10)  # arbitrary, random int from 1 to 10
    arg_used_mnemonics = [MagicMock()] * num_used_mnemonics

    fake_kalman_results = [MagicMock()] * pytest.gen.randint(
        1, 10
    )  # arbitrary, random int from 1 to 10
    len_fake_list = pytest.gen.randint(1, 10)  # arbitrary, random int from 1 to 10
    fake_kalman_results[0] = [MagicMock()] * len_fake_list

    cut = Diagnosis.__new__(Diagnosis)
    cut.has_kalman = True
    cut.kalman_results = fake_kalman_results

    mocker.patch(diagnosis.__name__ + ".copy.deepcopy")

    # Act
    result = cut.walkdown(arg_mnemonic_name, arg_used_mnemonics)

    # Assert
    assert result == Diagnosis.NO_DIAGNOSIS
    assert diagnosis.copy.deepcopy.call_count == 0
