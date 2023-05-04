# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

""" Test Data Conversion Functionality """
import pytest
from mock import MagicMock

import src.util.data_conversion

from numpy import ndarray

# floatify_input tests
def test_data_conversion_flotify_input_returns_empty_list_when_given__input_is_vacant(mocker):
    # Arrange
    arg__input = [] # empty list, no iterations
    arg_remove_str = False

    # Act
    result = src.util.data_conversion.floatify_input(arg__input, arg_remove_str)

    # Assert
    assert result == []

def test_data_conversion_flotify_input_raises_exception_when_float_returns_non_ValueError_exception(mocker):
    # Arrange
    arg__input = [str(MagicMock())] # list of single str list, 1 iteration
    arg_remove_str = False

    exception_message = str(MagicMock())
    fake_exception = Exception(exception_message)

    mocker.patch('builtins.float', side_effect=[fake_exception])

    # Act
    with pytest.raises(Exception) as e_info:
        src.util.data_conversion.floatify_input(arg__input, arg_remove_str)

    # Assert
    assert e_info.match(exception_message)

def test_data_conversion_flotify_input_returns_list_of_size_one_that_contains_the_call_to_float_when_no_Exception_is_thrown_and_given__input_is_str(mocker):
    # Arrange
    arg__input = []
    arg_remove_str = False 

    fake_item = str(MagicMock())
    arg__input.append(fake_item) # list of single str, one iteration

    expected_result = MagicMock()

    mocker.patch('builtins.float', return_value=expected_result)

    # Act
    result = src.util.data_conversion.floatify_input(arg__input, arg_remove_str)

    # Assert
    assert float.call_count == 1
    assert float.call_args_list[0].args == (arg__input[0], )
    assert result == [expected_result]

def test_data_conversion_flotify_input_returns_list_of_size_one_that_contains_the_second_call_to_float_after_replace_call_when_single_Exception_is_thrown(mocker):
    # Arrange
    arg__input = []
    arg_remove_str = False

    fake_item = MagicMock()
    arg__input.append(fake_item) # list of one item, one iteration

    expected_replaced_i = MagicMock()
    expected_result = MagicMock()

    mocker.patch('src.util.data_conversion.type', return_value=str)
    mocker.patch('src.util.data_conversion.float', side_effect=[ValueError, expected_result])
    mocker.patch.object(fake_item, 'replace', side_effect=[fake_item, fake_item, expected_replaced_i])
    
    # Act
    result = src.util.data_conversion.floatify_input(arg__input, arg_remove_str)

    # Assert
    assert src.util.data_conversion.float.call_count == 2
    assert src.util.data_conversion.float.call_args_list[0].args == (arg__input[0], )
    assert src.util.data_conversion.float.call_args_list[1].args == (expected_replaced_i, )
    assert fake_item.replace.call_count == 3
    assert fake_item.replace.call_args_list[0].args == ('-', '', )
    assert fake_item.replace.call_args_list[1].args == (':', '', )
    assert fake_item.replace.call_args_list[2].args == ('.', '', )
    assert result == [expected_result]

def test_data_conversion_flotify_input_returns_list_of_size_one_that_contains_0_dot_0_when_two_Exceptions_are_thrown_and_remove_str_is_False(mocker):
    # Arrange
    arg__input = []
    arg_remove_str = False

    fake_item = MagicMock()
    arg__input.append(fake_item) # list of one item, one iteration

    expected_replaced_i = MagicMock()

    mocker.patch('src.util.data_conversion.type', return_value=str)
    mocker.patch('src.util.data_conversion.float', side_effect=[ValueError, Exception])
    mocker.patch.object(fake_item, 'replace', side_effect=[fake_item, fake_item, expected_replaced_i])
    
    # Act
    result = src.util.data_conversion.floatify_input(arg__input, arg_remove_str)

    # Assert
    assert src.util.data_conversion.float.call_count == 2
    assert src.util.data_conversion.float.call_args_list[0].args == (arg__input[0], )
    assert src.util.data_conversion.float.call_args_list[1].args == (expected_replaced_i, )
    assert fake_item.replace.call_count == 3
    assert fake_item.replace.call_args_list[0].args == ('-', '', )
    assert fake_item.replace.call_args_list[1].args == (':', '', )
    assert fake_item.replace.call_args_list[2].args == ('.', '', )
    assert result == [0.0]

def test_data_conversion_flotify_input_default_arg_remove_str_is_False(mocker):
    # Arrange
    arg__input = []

    fake_item = MagicMock()
    arg__input.append(fake_item) # list of one item, one iteration

    expected_replaced_i = MagicMock()

    mocker.patch('src.util.data_conversion.type', return_value=str)
    mocker.patch('src.util.data_conversion.float', side_effect=[ValueError, Exception])
    mocker.patch.object(fake_item, 'replace', side_effect=[fake_item, fake_item, expected_replaced_i])
    
    # Act
    result = src.util.data_conversion.floatify_input(arg__input)

    # Assert
    assert result == [0.0] # shows flow was correct for remove_str being False

def test_data_conversion_flotify_input_returns_empty_list_when_two_Exceptions_are_thrown_and_remove_str_is_True(mocker):
    # Arrange
    arg__input = []
    arg_remove_str = True

    fake_item = MagicMock()
    arg__input.append(fake_item) # list of one item, one iteration

    expected_replaced_i = MagicMock()

    mocker.patch('src.util.data_conversion.type', return_value=str)
    mocker.patch('src.util.data_conversion.float', side_effect=[ValueError, Exception])
    mocker.patch.object(fake_item, 'replace', side_effect=[fake_item, fake_item, expected_replaced_i])
    
    # Act
    result = src.util.data_conversion.floatify_input(arg__input, arg_remove_str)

    # Assert
    assert src.util.data_conversion.float.call_count == 2
    assert src.util.data_conversion.float.call_args_list[0].args == (arg__input[0], )
    assert src.util.data_conversion.float.call_args_list[1].args == (expected_replaced_i, )
    assert fake_item.replace.call_count == 3
    assert fake_item.replace.call_args_list[0].args == ('-', '', )
    assert fake_item.replace.call_args_list[1].args == (':', '', )
    assert fake_item.replace.call_args_list[2].args == ('.', '', )
    assert result == []

def test_data_conversion_flotify_input_returns_call_to_float_that_was_given___input_item_when_type_of_item_is_not_str_and_there_is_single_item(mocker):
    # Arrange
    arg__input = []

    fake_item = MagicMock()
    arg__input.append(fake_item) # list of one item, one iteration

    expected_replaced_i = MagicMock()
    expected_result = MagicMock()

    mocker.patch('src.util.data_conversion.float', return_value=expected_result)
    mocker.patch.object(fake_item, 'replace', side_effect=[fake_item, fake_item, expected_replaced_i])
    
    # Act
    result = src.util.data_conversion.floatify_input(arg__input)

    # Assert
    assert result == [expected_result] # shows flow was correct for remove_str being False

def test_data_conversion_flotify_input_returns_expected_values_for_given__input_that_is_multi_typed_when_remove_str_is_True(mocker):
    # Arrange
    arg__input = []
    arg_remove_str = True

    side_effects_for_type = []
    side_effects_for_float = []
    expected_result = []

    num_fakes = pytest.gen.randint(0, 10) # arbitrary, from 0 to 10

    for i in range(num_fakes):
        rand_type_of_item = pytest.gen.sample(['str', 'str_need_replace', 'str_fail_replace', 'other'], 1)[0]

        if rand_type_of_item == 'str':
            arg__input.append(MagicMock())
            side_effects_for_type.append(str)
            resultant_float = MagicMock()
            side_effects_for_float.append(resultant_float)
            expected_result.append(resultant_float)
        elif rand_type_of_item == 'str_need_replace':
            fake_input = MagicMock()
            arg__input.append(fake_input)
            side_effects_for_type.append(str)
            resultant_float = MagicMock()
            side_effects_for_float.append(ValueError)
            mocker.patch.object(fake_input, 'replace', side_effect=[fake_input, fake_input, fake_input])
            side_effects_for_float.append(resultant_float)
            expected_result.append(resultant_float)
        elif rand_type_of_item == 'str_fail_replace':
            fake_input = MagicMock()
            arg__input.append(fake_input)
            side_effects_for_type.append(str)
            resultant_float = MagicMock()
            side_effects_for_float.append(ValueError)
            mocker.patch.object(fake_input, 'replace', side_effect=[fake_input, fake_input, fake_input])
            side_effects_for_float.append(Exception)
        else:
            arg__input.append(MagicMock())
            side_effects_for_type.append(False)
            resultant_float = MagicMock()
            side_effects_for_float.append(resultant_float)
            expected_result.append(resultant_float)

    mocker.patch('src.util.data_conversion.type', side_effect=side_effects_for_type)
    mocker.patch('src.util.data_conversion.float', side_effect=side_effects_for_float)
    
    # Act
    result = src.util.data_conversion.floatify_input(arg__input, arg_remove_str)

    # Assert
    assert result == expected_result
    
def test_data_conversion_flotify_input_returns_expected_values_for_given__input_that_is_multi_typed_when_remove_str_is_False(mocker):
    # Arrange
    arg__input = []
    arg_remove_str = False

    side_effects_for_type = []
    side_effects_for_float = []
    expected_result = []

    num_fakes = pytest.gen.randint(0, 10) # arbitrary, from 0 to 10

    for i in range(num_fakes):
        rand_type_of_item = pytest.gen.sample(['str', 'str_need_replace', 'str_fail_replace', 'other'], 1)[0]

        if rand_type_of_item == 'str':
            arg__input.append(MagicMock())
            side_effects_for_type.append(str)
            resultant_float = MagicMock()
            side_effects_for_float.append(resultant_float)
            expected_result.append(resultant_float)
        elif rand_type_of_item == 'str_need_replace':
            fake_input = MagicMock()
            arg__input.append(fake_input)
            side_effects_for_type.append(str)
            resultant_float = MagicMock()
            side_effects_for_float.append(ValueError)
            mocker.patch.object(fake_input, 'replace', side_effect=[fake_input, fake_input, fake_input])
            side_effects_for_float.append(resultant_float)
            expected_result.append(resultant_float)
        elif rand_type_of_item == 'str_fail_replace':
            fake_input = MagicMock()
            arg__input.append(fake_input)
            side_effects_for_type.append(str)
            resultant_float = MagicMock()
            side_effects_for_float.append(ValueError)
            mocker.patch.object(fake_input, 'replace', side_effect=[fake_input, fake_input, fake_input])
            side_effects_for_float.append(Exception)
            expected_result.append(0.0)
        else: # other
            arg__input.append(MagicMock())
            side_effects_for_type.append(False)
            resultant_float = MagicMock()
            side_effects_for_float.append(resultant_float)
            expected_result.append(resultant_float)

    mocker.patch('src.util.data_conversion.type', side_effect=side_effects_for_type)
    mocker.patch('src.util.data_conversion.float', side_effect=side_effects_for_float)
    
    # Act
    result = src.util.data_conversion.floatify_input(arg__input, arg_remove_str)

    # Assert
    assert result == expected_result
    
# status_to_oneHot tests
def test_data_conversion_status_to_oneHot_returns_given_status_when_status_isinstance_of_np_ndarray(mocker):
    # Arrange
    arg_status = MagicMock()

    mocker.patch('src.util.data_conversion.isinstance', return_value=True)

    # Act
    result = src.util.data_conversion.status_to_oneHot(arg_status)

    # Assert
    assert src.util.data_conversion.isinstance.call_count == 1
    assert src.util.data_conversion.isinstance.call_args_list[0].args == (arg_status, ndarray)
    assert result == arg_status

def test_data_conversion_status_to_oneHot_returns_one_hot_set_to_list_of_four_zeros_and_the_value_of_the_classes_status_to_1_point_0(mocker):
    # Arrange
    arg_status = MagicMock()

    fake_status = pytest.gen.randint(0,3) # size of array choice, from 0 to 3

    expected_result = [0.0, 0.0, 0.0, 0.0]
    expected_result[fake_status] = 1.0

    src.util.data_conversion.classes = {arg_status: fake_status}

    mocker.patch('src.util.data_conversion.isinstance', return_value=False)

    # Act
    result = src.util.data_conversion.status_to_oneHot(arg_status)

    # Assert
    assert result == expected_result

