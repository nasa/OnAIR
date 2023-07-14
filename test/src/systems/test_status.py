# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

""" Test Status Functionality """
import pytest
from mock import MagicMock

from src.systems.status import Status

# tests for init
def test_Status__init__with_empty_args_initializes_name_and_calls_set_status_with_default_values(mocker):
    # Arrange
    cut = Status.__new__(Status)

    mocker.patch.object(cut, 'set_status')

    # Act
    cut.__init__()

    # Assert
    assert cut.name == 'MISSION'
    assert cut.set_status.call_count == 1
    assert cut.set_status.call_args_list[0].args == ('---', -1.0)

def test_Status__init__with_valid_args_initializes_name_and_calls_set_status_with_expected_values(mocker):
    # Arrange
    cut = Status.__new__(Status)
    arg_name = MagicMock()
    arg_status = MagicMock()
    arg_bayesian_conf = MagicMock()

    mocker.patch.object(cut, 'set_status')

    # Act
    cut.__init__(arg_name, arg_status, arg_bayesian_conf)

    # Assert
    assert cut.name == arg_name
    assert cut.set_status.call_count == 1
    assert cut.set_status.call_args_list[0].args == (arg_status, arg_bayesian_conf)

# tests for set status
def test_Status_set_status_when_both_args_are_provided_and_valid_sets_variables_to_expected_values():
    # Arrange
    rand_index = pytest.gen.randint(0, 3) # index, from 0 to 3
    valid_statuses = ['---', 'RED', 'YELLOW', 'GREEN']
    arg_status = valid_statuses[rand_index]
    arg_bayesian_conf = pytest.gen.uniform(-1.0, 1.0) # float in accepted range from -1.0 to 1.0

    cut = Status.__new__(Status)

    # Act
    cut.set_status(arg_status, arg_bayesian_conf)

    # Assert
    assert cut.bayesian_conf == arg_bayesian_conf
    assert cut.status == arg_status

def test_Status_set_status_when_arg_status_is_valid_and_arg_conf_is_1_sets_variables_to_expected_values():
    # Arrange
    arg_status = '---'
    arg_bayesian_conf = 1.0

    cut = Status.__new__(Status)

    # Act
    cut.set_status(arg_status, arg_bayesian_conf)

    # Assert
    assert cut.bayesian_conf == arg_bayesian_conf
    assert cut.status == arg_status

def test_Status_set_status_when_arg_status_is_valid_and_arg_conf_is_negative_1_sets_variables_to_expected_values():
    # Arrange
    arg_status = '---'
    arg_bayesian_conf = -1.0

    cut = Status.__new__(Status)

    # Act
    cut.set_status(arg_status, arg_bayesian_conf)

    # Assert
    assert cut.bayesian_conf == arg_bayesian_conf
    assert cut.status == arg_status

def test_Status_set_status_when_only_stat_arg_is_provided_sets_variables_to_expected_values():
    # Arrange
    arg_status = '---'

    cut = Status.__new__(Status)

    # Act
    cut.set_status(arg_status)

    # Assert
    assert cut.bayesian_conf == -1.0
    assert cut.status == arg_status

def test_Status_set_status_raises_error_because_bayesian_conf_greater_than_1():
    # Arrange
    cut = Status.__new__(Status)
    
    arg_status = '---'
    arg_bayesian_conf = pytest.gen.uniform(1.01, 10.0) # arbitrary float greater than 1.0 (top of accepted range) 

    # Act
    with pytest.raises(AssertionError) as e_info:
        cut.set_status(arg_status, arg_bayesian_conf)

    # Assert
    assert e_info.match('')

def test_Status_set_status_raises_error_because_bayesian_conf_less_than_neg_1():
    # Arrange
    cut = Status.__new__(Status)
    
    arg_status = '---'
    arg_bayesian_conf = pytest.gen.uniform(-10.0, -1.01) # arbitrary float less than -1.0 (bottom of accepted range)

    # Act
    with pytest.raises(AssertionError) as e_info:
        cut.set_status(arg_status, arg_bayesian_conf)

    # Assert
    assert e_info.match('')

def test_Status_set_status_raises_error_because_invalid_status_arg():
    # Arrange
    cut = Status.__new__(Status)
    arg_status = str(MagicMock())
    arg_bayesian_conf = 0

    # Act
    with pytest.raises(AssertionError) as e_info:
        cut.set_status(arg_status, arg_bayesian_conf)

    # Assert
    assert e_info.match('')

# tests for get_status
def test_Status_get_status_returns_expected_values():
    # Arrange
    fake_status = MagicMock()
    cut = Status.__new__(Status)
    cut.status = fake_status

    # Act
    result = cut.get_status()

    # Assert
    assert result == fake_status

# tests for get_bayesian_status
def test_Status_get_bayesian_status_returns_expected_values():
    # Arrange
    fake_status = MagicMock()
    fake_bayesian_conf = MagicMock()

    cut = Status.__new__(Status)
    cut.status = fake_status
    cut.bayesian_conf = fake_bayesian_conf

    # Act
    result_status, result_bayesian_conf = cut.get_bayesian_status()

    # Assert
    assert result_status == fake_status
    assert result_bayesian_conf == fake_bayesian_conf

# tests for get_name
def test_Status_get_name_returns_expected_value():
    # Arrange
    fake_name = MagicMock()
    cut = Status.__new__(Status)
    cut.name = fake_name

    # Act
    result = cut.get_name()

    # Assert
    assert result == fake_name
