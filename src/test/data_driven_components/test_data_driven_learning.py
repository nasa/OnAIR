""" Test DataDrivenLearning Functionality """
import pytest
from mock import MagicMock
import src.data_driven_components.data_driven_learning as data_driven_learning
from src.data_driven_components.data_driven_learning import DataDrivenLearning

from numpy import ndarray

# __init__ tests
def test__init__sets_the_expected_classes_and_calls_init_learning_systems_with_given_headers_and_sample_input_but_does_not_itself_set_headers_when_there_is_no_Exception(mocker):
    # Arrange
    arg_headers = MagicMock()
    arg_sample_input = MagicMock()

    expected_classes = {'RED' : 0,
                     'YELLOW' : 1,
                      'GREEN' : 2,
                        '---' : 3}
    expected_inverted_classes = {0 : 'RED',
                                 1 : 'YELLOW',
                                 2 : 'GREEN',
                                 3 : '---'}

    cut = DataDrivenLearning.__new__(DataDrivenLearning)

    mocker.patch.object(cut, 'init_learning_systems')

    # Act
    cut.__init__(arg_headers, arg_sample_input)

    # Assert
    assert cut.init_learning_systems.call_count == 1
    assert cut.init_learning_systems.call_args_list[0].args == (arg_headers, arg_sample_input, )
    assert cut.classes == expected_classes
    assert cut.inverted_classes == expected_inverted_classes
    assert hasattr(cut, 'headers') == False

def test__init__sets_the_expected_classes_and_call_to_init_learning_systems_raises_Exception_so_headers_is_set_to_empty_list(mocker):
    # Arrange
    arg_headers = MagicMock()
    arg_sample_input = MagicMock()

    expected_classes = {'RED' : 0,
                     'YELLOW' : 1,
                      'GREEN' : 2,
                        '---' : 3}
    expected_inverted_classes = {0 : 'RED',
                                 1 : 'YELLOW',
                                 2 : 'GREEN',
                                 3 : '---'}

    cut = DataDrivenLearning.__new__(DataDrivenLearning)

    mocker.patch.object(cut, 'init_learning_systems', side_effect=Exception())

    # Act
    cut.__init__(arg_headers, arg_sample_input)

    # Assert
    assert cut.init_learning_systems.call_count == 1
    assert cut.init_learning_systems.call_args_list[0].args == (arg_headers, arg_sample_input, )
    assert cut.classes == expected_classes
    assert cut.inverted_classes == expected_inverted_classes
    assert cut.headers == []

def test__init__default_value_for_each_argument_is_empty_list(mocker):
    # Arrange
    cut = DataDrivenLearning.__new__(DataDrivenLearning)

    mocker.patch.object(cut, 'init_learning_systems', side_effect={Exception() if pytest.gen.randint(0,1) else ''})

    # Act
    cut.__init__()

    # Assert
    assert cut.init_learning_systems.call_count == 1
    assert cut.init_learning_systems.call_args_list[0].args == ([], [], )

# init_learning_systems tests
def test_init_learning_systems_asserts_when_headers_length_is_0(mocker):
    # Arrange
    arg_headers = []

    cut = DataDrivenLearning.__new__(DataDrivenLearning)

    # Act
    with pytest.raises(AssertionError) as e_info:
        cut.init_learning_systems(arg_headers)
    
    # Assert - taken care of by pytest.raises

def test_init_learning_systems_returns_tuple_of_sample_input_set_to_all_zeros_list_equal_in_size_to_given_headers_len_and_sample_output_set_to_call_to_status_to_oneHot_when_given_sample_is_default(mocker):
    # Arrange
    arg_headers = []
    
    num_headers = pytest.gen.randint(1, 10) # arbitrary from 1 to 10 items
    expected_sample_input = []
    
    for i in range(num_headers): 
        arg_headers.append(MagicMock())
        expected_sample_input.append(0.0)

    expected_sample_output = MagicMock()

    cut = DataDrivenLearning.__new__(DataDrivenLearning)

    mocker.patch.object(cut, 'floatify_input')
    mocker.patch.object(cut, 'status_to_oneHot', return_value=expected_sample_output)

    # Act
    result = cut.init_learning_systems(arg_headers)

    # Assert
    assert cut.status_to_oneHot.call_count == 1
    assert cut.status_to_oneHot.call_args_list[0].args == ('---', )
    assert result == (expected_sample_input, expected_sample_output)

def test_init_learning_systems_returns_tuple_of_sample_input_set_to_all_zeros_list_equal_in_size_to_given_headers_len_and_sample_output_set_to_call_to_status_to_oneHot_when_given_sample_is_vacant(mocker):
    # Arrange
    arg_headers = []
    arg_sample = []
    
    num_headers = pytest.gen.randint(1, 10) # arbitrary from 1 to 10 items
    expected_sample_input = []
    
    for i in range(num_headers): 
        arg_headers.append(MagicMock())
        expected_sample_input.append(0.0)

    expected_sample_output = MagicMock()

    cut = DataDrivenLearning.__new__(DataDrivenLearning)

    mocker.patch.object(cut, 'floatify_input')
    mocker.patch.object(cut, 'status_to_oneHot', return_value=expected_sample_output)

    # Act
    result = cut.init_learning_systems(arg_headers, arg_sample)

    # Assert
    assert cut.floatify_input.call_count == 0
    assert cut.status_to_oneHot.call_count == 1
    assert cut.status_to_oneHot.call_args_list[0].args == ('---', )
    assert result == (expected_sample_input, expected_sample_output)

def test_init_learning_systems_returns_tuple_of_sample_input_set_to_call_to_floatify_input_and_sample_output_set_to_call_to_status_to_oneHot_when_given_sample_is_occupied(mocker):
    # Arrange
    arg_headers = [MagicMock()]
    arg_sample = [MagicMock()]
    
    expected_sample_input = MagicMock()
    expected_sample_output = MagicMock()

    cut = DataDrivenLearning.__new__(DataDrivenLearning)

    mocker.patch.object(cut, 'floatify_input', return_value=expected_sample_input)
    mocker.patch.object(cut, 'status_to_oneHot', return_value=expected_sample_output)

    # Act
    result = cut.init_learning_systems(arg_headers, arg_sample)

    # Assert
    assert cut.floatify_input.call_count == 1
    assert cut.floatify_input.call_args_list[0].args == (arg_sample, )
    assert cut.status_to_oneHot.call_count == 1
    assert cut.status_to_oneHot.call_args_list[0].args == ('---', )
    assert result == (expected_sample_input, expected_sample_output)

def test_init_learning_systems_sets_self_headers_to_given_headers(mocker):
    # Arrange
    arg_headers = [MagicMock()]

    cut = DataDrivenLearning.__new__(DataDrivenLearning)

    mocker.patch.object(cut, 'status_to_oneHot')

    # Act
    cut.init_learning_systems(arg_headers)

    # Assert
    assert cut.headers == arg_headers

# update tests
def test_update_returns_tuple_of_call_to_floatify_input_and_call_to_status_to_oneHot(mocker):
    # Arrange
    arg_curr_data = MagicMock()
    arg_status = MagicMock()

    expected_input_data = MagicMock()
    expected_output_data = MagicMock()

    cut = DataDrivenLearning.__new__(DataDrivenLearning)

    mocker.patch.object(cut, 'floatify_input', return_value=expected_input_data)
    mocker.patch.object(cut, 'status_to_oneHot', return_value=expected_output_data)

    # Act
    result = cut.update(arg_curr_data, arg_status)

    # Assert
    assert cut.floatify_input.call_count == 1
    assert cut.floatify_input.call_args_list[0].args == (arg_curr_data, )
    assert cut.status_to_oneHot.call_count == 1
    assert cut.status_to_oneHot.call_args_list[0].args == (arg_status, )
    assert result == (expected_input_data, expected_output_data)

# apriori_training tests
def test_apriori_training_does_nothing():
    # Arrange
    arg_batch_data = MagicMock()

    cut = DataDrivenLearning.__new__(DataDrivenLearning)

    # Act
    result = cut.apriori_training(arg_batch_data)

    # Assert
    assert result == None

# floatify_input tests
def test_flotify_input_returns_empty_list_when_given__input_is_vacant(mocker):
    # Arrange
    arg__input = [] # empty list, no iterations

    cut = DataDrivenLearning.__new__(DataDrivenLearning)

    # Act
    result = cut.floatify_input(arg__input)

    # Assert
    assert result == []

def test_flotify_input_returns_list_of_size_one_that_contains_the_call_to_float_when_no_Exception_is_thrown_and_given__input_is_str(mocker):
    # Arrange
    arg__input = [] 

    fake_item = str(MagicMock())
    arg__input.append(fake_item) # list of single str, one iteration

    expected_result = MagicMock()

    cut = DataDrivenLearning.__new__(DataDrivenLearning)

    mocker.patch('src.data_driven_components.data_driven_learning.float', return_value=expected_result)

    # Act
    result = cut.floatify_input(arg__input)

    # Assert
    assert data_driven_learning.float.call_count == 1
    assert data_driven_learning.float.call_args_list[0].args == (arg__input[0], )
    assert result == [expected_result]

def test_flotify_input_returns_list_of_size_one_that_contains_the_second_call_to_float_after_replace_call_when_single_Exception_is_thrown(mocker):
    # Arrange
    arg__input = []

    fake_item = MagicMock()
    arg__input.append(fake_item) # list of one item, one iteration

    expected_replaced_i = MagicMock()
    expected_result = MagicMock()

    cut = DataDrivenLearning.__new__(DataDrivenLearning)

    mocker.patch('src.data_driven_components.data_driven_learning.type', return_value=str)
    mocker.patch('src.data_driven_components.data_driven_learning.float', side_effect=[Exception, expected_result])
    mocker.patch.object(fake_item, 'replace', side_effect=[fake_item, fake_item, expected_replaced_i])
    
    # Act
    result = cut.floatify_input(arg__input)

    # Assert
    assert data_driven_learning.float.call_count == 2
    assert data_driven_learning.float.call_args_list[0].args == (arg__input[0], )
    assert data_driven_learning.float.call_args_list[1].args == (expected_replaced_i, )
    assert fake_item.replace.call_count == 3
    assert fake_item.replace.call_args_list[0].args == ('-', '', )
    assert fake_item.replace.call_args_list[1].args == (':', '', )
    assert fake_item.replace.call_args_list[2].args == ('.', '', )
    assert result == [expected_result]

def test_flotify_input_returns_list_of_size_one_that_contains_0_dot_0_when_two_Exceptions_are_thrown_and_remove_str_is_False(mocker):
    # Arrange
    arg__input = []
    arg_remove_str = False

    fake_item = MagicMock()
    arg__input.append(fake_item) # list of one item, one iteration

    expected_replaced_i = MagicMock()

    cut = DataDrivenLearning.__new__(DataDrivenLearning)

    mocker.patch('src.data_driven_components.data_driven_learning.type', return_value=str)
    mocker.patch('src.data_driven_components.data_driven_learning.float', side_effect=[Exception, Exception])
    mocker.patch.object(fake_item, 'replace', side_effect=[fake_item, fake_item, expected_replaced_i])
    
    # Act
    result = cut.floatify_input(arg__input, arg_remove_str)

    # Assert
    assert data_driven_learning.float.call_count == 2
    assert data_driven_learning.float.call_args_list[0].args == (arg__input[0], )
    assert data_driven_learning.float.call_args_list[1].args == (expected_replaced_i, )
    assert fake_item.replace.call_count == 3
    assert fake_item.replace.call_args_list[0].args == ('-', '', )
    assert fake_item.replace.call_args_list[1].args == (':', '', )
    assert fake_item.replace.call_args_list[2].args == ('.', '', )
    assert result == [0.0]

def test_flotify_input_default_arg_remove_str_is_False(mocker):
    # Arrange
    arg__input = []

    fake_item = MagicMock()
    arg__input.append(fake_item) # list of one item, one iteration

    expected_replaced_i = MagicMock()

    cut = DataDrivenLearning.__new__(DataDrivenLearning)

    mocker.patch('src.data_driven_components.data_driven_learning.type', return_value=str)
    mocker.patch('src.data_driven_components.data_driven_learning.float', side_effect=[Exception, Exception])
    mocker.patch.object(fake_item, 'replace', side_effect=[fake_item, fake_item, expected_replaced_i])
    
    # Act
    result = cut.floatify_input(arg__input)

    # Assert
    assert result == [0.0] # shows flow was correct for remove_str being False

def test_flotify_input_returns_empty_list_when_two_Exceptions_are_thrown_and_remove_str_is_True(mocker):
    # Arrange
    arg__input = []
    arg_remove_str = True

    fake_item = MagicMock()
    arg__input.append(fake_item) # list of one item, one iteration

    expected_replaced_i = MagicMock()

    cut = DataDrivenLearning.__new__(DataDrivenLearning)

    mocker.patch('src.data_driven_components.data_driven_learning.type', return_value=str)
    mocker.patch('src.data_driven_components.data_driven_learning.float', side_effect=[Exception, Exception])
    mocker.patch.object(fake_item, 'replace', side_effect=[fake_item, fake_item, expected_replaced_i])
    
    # Act
    result = cut.floatify_input(arg__input, arg_remove_str)

    # Assert
    assert data_driven_learning.float.call_count == 2
    assert data_driven_learning.float.call_args_list[0].args == (arg__input[0], )
    assert data_driven_learning.float.call_args_list[1].args == (expected_replaced_i, )
    assert fake_item.replace.call_count == 3
    assert fake_item.replace.call_args_list[0].args == ('-', '', )
    assert fake_item.replace.call_args_list[1].args == (':', '', )
    assert fake_item.replace.call_args_list[2].args == ('.', '', )
    assert result == []

def test_flotify_input_returns_call_to_float_that_was_given___input_item_when_type_of_item_is_not_str_and_there_is_single_item(mocker):
    # Arrange
    arg__input = []

    fake_item = MagicMock()
    arg__input.append(fake_item) # list of one item, one iteration

    expected_replaced_i = MagicMock()
    expected_result = MagicMock()

    cut = DataDrivenLearning.__new__(DataDrivenLearning)

    mocker.patch('src.data_driven_components.data_driven_learning.float', return_value=expected_result)
    mocker.patch.object(fake_item, 'replace', side_effect=[fake_item, fake_item, expected_replaced_i])
    
    # Act
    result = cut.floatify_input(arg__input)

    # Assert
    assert result == [expected_result] # shows flow was correct for remove_str being False

def test_flotify_input_returns_expected_values_for_given__input_that_is_multi_typed_when_remove_str_is_True(mocker):
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
            side_effects_for_float.append(Exception)
            mocker.patch.object(fake_input, 'replace', side_effect=[fake_input, fake_input, fake_input])
            side_effects_for_float.append(resultant_float)
            expected_result.append(resultant_float)
        elif rand_type_of_item == 'str_fail_replace':
            fake_input = MagicMock()
            arg__input.append(fake_input)
            side_effects_for_type.append(str)
            resultant_float = MagicMock()
            side_effects_for_float.append(Exception)
            mocker.patch.object(fake_input, 'replace', side_effect=[fake_input, fake_input, fake_input])
            side_effects_for_float.append(Exception)
        else:
            arg__input.append(MagicMock())
            side_effects_for_type.append(False)
            resultant_float = MagicMock()
            side_effects_for_float.append(resultant_float)
            expected_result.append(resultant_float)

    cut = DataDrivenLearning.__new__(DataDrivenLearning)

    mocker.patch('src.data_driven_components.data_driven_learning.type', side_effect=side_effects_for_type)
    mocker.patch('src.data_driven_components.data_driven_learning.float', side_effect=side_effects_for_float)

    # Act
    result = cut.floatify_input(arg__input, arg_remove_str)

    # Assert
    assert result == expected_result
    
def test_flotify_input_returns_expected_values_for_given__input_that_is_multi_typed_when_remove_str_is_False(mocker):
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
            side_effects_for_float.append(Exception)
            mocker.patch.object(fake_input, 'replace', side_effect=[fake_input, fake_input, fake_input])
            side_effects_for_float.append(resultant_float)
            expected_result.append(resultant_float)
        elif rand_type_of_item == 'str_fail_replace':
            fake_input = MagicMock()
            arg__input.append(fake_input)
            side_effects_for_type.append(str)
            resultant_float = MagicMock()
            side_effects_for_float.append(Exception)
            mocker.patch.object(fake_input, 'replace', side_effect=[fake_input, fake_input, fake_input])
            side_effects_for_float.append(Exception)
            expected_result.append(0.0)
        else: # other
            arg__input.append(MagicMock())
            side_effects_for_type.append(False)
            resultant_float = MagicMock()
            side_effects_for_float.append(resultant_float)
            expected_result.append(resultant_float)

    cut = DataDrivenLearning.__new__(DataDrivenLearning)

    mocker.patch('src.data_driven_components.data_driven_learning.type', side_effect=side_effects_for_type)
    mocker.patch('src.data_driven_components.data_driven_learning.float', side_effect=side_effects_for_float)

    # Act
    result = cut.floatify_input(arg__input, arg_remove_str)

    # Assert
    assert result == expected_result
    
# status_to_oneHot tests
def test_status_to_oneHot_returns_given_status_when_status_isinstance_of_np_ndarray(mocker):
    # Arrange
    arg_status = MagicMock()

    cut = DataDrivenLearning.__new__(DataDrivenLearning)

    mocker.patch('src.data_driven_components.data_driven_learning.isinstance', return_value=True)

    # Act
    result = cut.status_to_oneHot(arg_status)

    # Assert
    assert data_driven_learning.isinstance.call_count == 1
    assert data_driven_learning.isinstance.call_args_list[0].args == (arg_status, ndarray)
    assert result == arg_status

def test_status_to_oneHot_returns_one_hot_set_to_list_of_four_zeros_and_the_value_of_the_classes_status_to_1_point_0(mocker):
    # Arrange
    arg_status = MagicMock()

    fake_status = pytest.gen.randint(0,3) # size of array choice, from 0 to 3

    expected_result = [0.0, 0.0, 0.0, 0.0]
    expected_result[fake_status] = 1.0

    cut = DataDrivenLearning.__new__(DataDrivenLearning)
    cut.classes = {arg_status: fake_status}

    mocker.patch('src.data_driven_components.data_driven_learning.isinstance', return_value=False)

    # Act
    result = cut.status_to_oneHot(arg_status)

    # Assert
    assert result == expected_result
