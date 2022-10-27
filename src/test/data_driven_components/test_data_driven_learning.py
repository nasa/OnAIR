""" Test DataDrivenLearning Functionality """
import pytest
from mock import MagicMock
import src.data_driven_components.data_driven_learning as data_driven_learning
from src.data_driven_components.data_driven_learning import DataDrivenLearning

# __init__ tests
def test_DataDrivenLearning__init__asserts_when_given_headers_is_empty(mocker):
    # Arrange
    arg_headers = []
    arg_AI_constructs = MagicMock()

    cut = DataDrivenLearning.__new__(DataDrivenLearning)

    # Act
    with pytest.raises(AssertionError) as e_info:
        cut.__init__(arg_headers, arg_AI_constructs)

    # Assert
    assert e_info.match('')
    assert hasattr(cut, 'headers') == False

def test_DataDrivenLearning__init__asserts_when_no_arguments_are_given_because_default_headers_is_empty_list(mocker):
    # Arrange

    cut = DataDrivenLearning.__new__(DataDrivenLearning)

    # Act
    with pytest.raises(AssertionError) as e_info:
        cut.__init__()

    # Assert
    assert e_info.match('')
    assert hasattr(cut, 'headers') == False

def test_DataDrivenLearning__init__sets_instance_headers_to_given_headers(mocker):
    # Arrange
    arg_headers = []
    arg_AI_constructs = MagicMock()

    num_fake_headers = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10 headers (0 has own test)
    for i in range(num_fake_headers):
        arg_headers.append(MagicMock())

    cut = DataDrivenLearning.__new__(DataDrivenLearning)

    # Act
    cut.__init__(arg_headers, arg_AI_constructs)

    # Assert
    assert cut.headers == arg_headers

def test_DataDrivenLearning__init__default_value_for_AI_constucts_is_empty_list_but_it_affects_nothing(mocker):
    # Arrange
    arg_headers = []

    num_fake_headers = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10 headers (0 has own test)
    for i in range(num_fake_headers):
        arg_headers.append(MagicMock())

    cut = DataDrivenLearning.__new__(DataDrivenLearning)

    # Act
    cut.__init__(arg_headers)

    # Assert
    assert cut.headers == arg_headers

# update tests
def test_DataDrivenLearning_update_returns_tuple_of_call_to_floatify_input_and_call_to_status_to_oneHot(mocker):
    # Arrange
    arg_curr_data = MagicMock()
    arg_status = MagicMock()

    expected_input_data = MagicMock()
    expected_output_data = MagicMock()

    cut = DataDrivenLearning.__new__(DataDrivenLearning)

    mocker.patch('src.data_driven_components.data_driven_learning.floatify_input', return_value=expected_input_data)
    mocker.patch('src.data_driven_components.data_driven_learning.status_to_oneHot', return_value=expected_output_data)

    # Act
    result = cut.update(arg_curr_data, arg_status)

    # Assert
    assert data_driven_learning.floatify_input.call_count == 1
    assert data_driven_learning.floatify_input.call_args_list[0].args == (arg_curr_data, )
    assert data_driven_learning.status_to_oneHot.call_count == 1
    assert data_driven_learning.status_to_oneHot.call_args_list[0].args == (arg_status, )
    assert result == (expected_input_data, expected_output_data)

# apriori_training tests
def test_DataDrivenLearning_apriori_training_does_nothing():
    # Arrange
    arg_batch_data = MagicMock()

    cut = DataDrivenLearning.__new__(DataDrivenLearning)

    # Act
    result = cut.apriori_training(arg_batch_data)

    # Assert
    assert result == None
