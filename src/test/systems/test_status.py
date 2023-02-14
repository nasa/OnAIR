""" Test Status Functionality """
import pytest
from mock import MagicMock

from src.systems.status import Status

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

# tests for set status
def test_Status_set_status_when_both_args_are_provided_sets_variables_to_expected_values():
    # Arrange
    arg_status = MagicMock()
    arg_bayesian_conf = MagicMock()

    cut = Status.__new__(Status)

    # Act
    cut.set_status(arg_status, arg_bayesian_conf)

    # Assert
    assert cut.bayesian_conf == arg_bayesian_conf
    assert cut.status == arg_status

def test_Status_set_status_when_only_stat_arg_is_provided_sets_variables_to_expected_values():
    # Arrange
    arg_status = MagicMock()

    cut = Status.__new__(Status)

    # Act
    cut.set_status(arg_status)

    # Assert
    assert cut.bayesian_conf == -1
    assert cut.status == arg_status

# tests for init
def test_Status__init__with_empty_args_initializes_variables_to_default_values():
    # Arrange
    cut = Status.__new__(Status)

    # Act
    cut.__init__()

    # Assert
    assert cut.name == 'MISSION'
    assert cut.bayesian_conf == -1
    assert cut.status == '---'

def test_Status__init__with_valid_args_initializes_variables_to_expected_values():
    # Arrange
    cut = Status.__new__(Status)
    arg_name = MagicMock()
    
    rand_index = pytest.gen.randint(0, 3) # index, from 0 to 3
    valid_statuses = ['---', 'RED', 'YELLOW', 'GREEN']
    arg_status = valid_statuses[rand_index]
    arg_bayesian_conf = pytest.gen.uniform(-1.0, 1.0) # float in accepted range from -1.0 to 1.0

    # Act
    cut.__init__(arg_name, arg_status, arg_bayesian_conf)

    # Assert
    assert cut.name == arg_name
    assert cut.bayesian_conf == arg_bayesian_conf
    assert cut.status == arg_status

def test_Status__init__accepts_bayesian_conf_arg_equal_to_1():
    # Arrange
    cut = Status.__new__(Status)
    arg_name = MagicMock()
    
    arg_status = '---'
    arg_bayesian_conf = 1.0

    # Act
    cut.__init__(arg_name, arg_status, arg_bayesian_conf)

    # Assert
    assert cut.name == arg_name
    assert cut.bayesian_conf == arg_bayesian_conf
    assert cut.status == arg_status

def test_Status__init__accepts_bayesian_conf_arg_equal_to_negative_1():
    # Arrange
    cut = Status.__new__(Status)
    arg_name = MagicMock()
    
    arg_status = '---'
    arg_bayesian_conf = -1.0

    # Act
    cut.__init__(arg_name, arg_status, arg_bayesian_conf)

    # Assert
    assert cut.name == arg_name
    assert cut.bayesian_conf == arg_bayesian_conf
    assert cut.status == arg_status

def test_Status__init__raises_error_because_bayesian_conf_greater_than_1():
    # Arrange
    cut = Status.__new__(Status)
    arg_name = MagicMock()
    
    arg_status = '---'
    arg_bayesian_conf = pytest.gen.uniform(1.01, 10.0) # arbitrary float greater than 1.0 (top of accepted range) 

    # Act
    with pytest.raises(AssertionError) as e_info:
        cut.__init__(arg_name, arg_status, arg_bayesian_conf)

    # Assert
    assert e_info.match('')

def test_Status__init__raises_error_because_bayesian_conf_less_than_neg_1():
    # Arrange
    cut = Status.__new__(Status)
    arg_name = MagicMock()
    
    arg_status = '---'
    arg_bayesian_conf = pytest.gen.uniform(-10.0, -1.01) # arbitrary float less than -1.0 (bottom of accepted range)

    # Act
    with pytest.raises(AssertionError) as e_info:
        cut.__init__(arg_name, arg_status, arg_bayesian_conf)

    # Assert
    assert e_info.match('')

def test_Status__init__raises_error_because_invalid_status_arg():
    # Arrange
    cut = Status.__new__(Status)
    arg_name = MagicMock()
    arg_status = str(MagicMock())
    arg_bayesian_conf = 0

    # Act
    with pytest.raises(AssertionError) as e_info:
        cut.__init__(arg_name, arg_status, arg_bayesian_conf)

    # Assert
    assert e_info.match('')
