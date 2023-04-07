""" Test Parser Util Functionality """
import pytest
from mock import MagicMock
import data_handling.parsers.tlm_json_parser as tlm_parser

# parseTlmConfJson tests


# reorganizeTlmDict tests
def test_reorganizeTlmDict_raises_error_when_arg_data_does_not_contain_subsystems_key():
    # Arrange
    arg_data_len = pytest.gen.randint(0, 10) # arbitrary, from 0 to 10
    arg_data = {}
    [arg_data.update({MagicMock() : MagicMock()}) for i in range(arg_data_len)]

    # Assert
    with pytest.raises(KeyError) as e_info:
        result = tlm_parser.reorganizeTlmDict(arg_data)

    # Act
    assert e_info.match('subsystems')

def test_reorganizeTlmDict_returns_empty_dict_when_arg_data_subsystems_exists_and_is_empty():
    # Arrange
    arg_data_len = pytest.gen.randint(0, 10) # arbitrary, from 0 to 10
    arg_data = {}
    [arg_data.update({MagicMock() : MagicMock()}) for i in range(arg_data_len)]
    arg_data.update({'subsystems' : {}})

    # Assert
    result = tlm_parser.reorganizeTlmDict(arg_data)

    # Act
    assert result == {}

def test_reorganizeTlmDict_updates_data_with_return_value_from_recursive_helper_function_when_arg_data_subsystems_exists_and_is_not_empty(mocker):
    # Arrange
    fake_app_name = MagicMock()
    fake_app_data = MagicMock()
    fake_app_dict = {fake_app_name : fake_app_data}
    num_subsystems = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10
    fake_subsystems = [MagicMock() for i in range(num_subsystems)]

    arg_data_len = pytest.gen.randint(0, 10) # arbitrary, from 0 to 10
    arg_data = {}
    [arg_data.update({MagicMock() : MagicMock()}) for i in range(arg_data_len)]
    arg_data.update({'subsystems' : {}})
    [arg_data['subsystems'].update({fake_ss : fake_app_dict}) for fake_ss in fake_subsystems]

    forced_side_effect_list = [{MagicMock() : MagicMock()} for i in range(num_subsystems)]
    mocker.patch('data_handling.parsers.tlm_json_parser.reorganizeTlmDictRecursiveStep', side_effect=forced_side_effect_list)

    expected_result = {}
    [expected_result.update(app_data) for app_data in forced_side_effect_list]

    # Assert
    result = tlm_parser.reorganizeTlmDict(arg_data)

    # Act
    assert result == expected_result
    assert tlm_parser.reorganizeTlmDictRecursiveStep.call_count == num_subsystems
    for i in range(num_subsystems):
        assert tlm_parser.reorganizeTlmDictRecursiveStep.call_args_list[i].args == (fake_app_name, fake_subsystems[i], fake_app_data)

# reorganizeTlmDictRecursiveStep tests


# str2lst tests
def test_tlm_json_parser_str2lst_returns_call_to_ast_literal_eval_which_receive_given_string(mocker):
    # Arrange
    arg_string = str(MagicMock())

    expected_result = MagicMock()
    
    mocker.patch('data_handling.parsers.tlm_json_parser.ast.literal_eval', return_value=expected_result)

    # Act
    result = tlm_parser.str2lst(arg_string)

    # Assert
    assert tlm_parser.ast.literal_eval.call_count == 1
    assert tlm_parser.ast.literal_eval.call_args_list[0].args == (arg_string, )
    assert result == expected_result

def test_tlm_json_parser_str2lst_prints_message_when_ast_literal_eval_receives_given_string_but_raises_exception(mocker):
    # Arrange
    arg_string = str(MagicMock())
    
    mocker.patch('data_handling.parsers.tlm_json_parser.ast.literal_eval', side_effect=Exception)
    mocker.patch('data_handling.parsers.tlm_json_parser.print')
    
    # Act
    result = tlm_parser.str2lst(arg_string)

    # Assert
    assert tlm_parser.ast.literal_eval.call_count == 1
    assert tlm_parser.ast.literal_eval.call_args_list[0].args == (arg_string, )
    assert tlm_parser.print.call_count == 1
    assert tlm_parser.print.call_args_list[0].args == ("Unable to process string representation of list", )
    assert result == None

# parseJson tests
def test_tlm_json_parser_parseJson_opens_given_path_and_returns_data_returned_by_orjson(mocker):
    # Arrange
    arg_path = MagicMock()

    fake_file = MagicMock()
    fake_file_str = MagicMock()
    fake_file_data = MagicMock()

    mocker.patch('data_handling.parsers.tlm_json_parser.open', return_value=fake_file)
    mocker.patch.object(fake_file, 'read', return_value=fake_file_str)
    mocker.patch('data_handling.parsers.tlm_json_parser.orjson.loads', return_value=fake_file_data)
    mocker.patch.object(fake_file, 'close')

    # Act
    result = tlm_parser.parseJson(arg_path)
    
    # Assert
    assert tlm_parser.open.call_count == 1
    assert tlm_parser.open.call_args_list[0].args == (arg_path, 'rb')
    assert fake_file.read.call_count == 1
    assert tlm_parser.orjson.loads.call_count == 1
    assert tlm_parser.orjson.loads.call_args_list[0].args == (fake_file_str, )
    assert fake_file.close.call_count == 1
    assert result == fake_file_data

# writeToJson tests
def test_tlm_json_parser_writeJson_opens_given_path_and_writes_data_using_orjson(mocker):
    # Arrange
    arg_path = MagicMock()
    arg_data = MagicMock()

    fake_file = MagicMock()
    fake_json_data = MagicMock()

    mocker.patch('data_handling.parsers.tlm_json_parser.open', return_value=fake_file)
    mocker.patch.object(fake_file, 'write')
    mocker.patch('data_handling.parsers.tlm_json_parser.orjson.dumps', return_value=fake_json_data)
    mocker.patch.object(fake_file, 'close')

    # Act
    tlm_parser.writeToJson(arg_path, arg_data)
    
    # Assert
    assert tlm_parser.open.call_count == 1
    assert tlm_parser.open.call_args_list[0].args == (arg_path, 'wb')
    assert tlm_parser.orjson.dumps.call_count == 1
    assert tlm_parser.orjson.dumps.call_args_list[0].args == (arg_data, )
    assert tlm_parser.orjson.dumps.call_args_list[0].kwargs == {'option' : tlm_parser.orjson.OPT_INDENT_2}
    assert fake_file.write.call_count == 1
    assert fake_file.write.call_args_list[0].args == (fake_json_data, )
    assert fake_file.close.call_count == 1

# ----- Tests for conversion functions -----

# convertTlmToJson tests


# convertTlmDictToJsonDict tests


# getJsonData tests


# parseTlmConfTxt tests


# getConfigPath tests
def test_tlm_json_parser_getConfigPath_uses_os_functions_to_find_file_path(mocker):
    # Arrange
    fake__file__ = MagicMock()
    fake_parent_dir = MagicMock()
    fake_data_dir = MagicMock()
    fake_configs_dir = MagicMock()
    fake_file_path = MagicMock()

    arg_file_name = MagicMock()

    mocker.patch('data_handling.parsers.tlm_json_parser.__file__', fake__file__)
    mocker.patch('data_handling.parsers.tlm_json_parser.os.path.dirname', return_value=fake_parent_dir)
    mocker.patch('data_handling.parsers.tlm_json_parser.os.path.join', side_effect=[fake_data_dir, fake_configs_dir, fake_file_path])

    # Act
    result = tlm_parser.getConfigPath(arg_file_name)

    # Assert
    assert result == fake_file_path
    assert tlm_parser.os.path.dirname.call_count == 3
    assert tlm_parser.os.path.dirname.call_args_list[0].args == (fake__file__, )
    assert tlm_parser.os.path.dirname.call_args_list[1].args == (fake_parent_dir, )
    assert tlm_parser.os.path.dirname.call_args_list[2].args == (fake_parent_dir, )
    assert tlm_parser.os.path.join.call_count == 3
    assert tlm_parser.os.path.join.call_args_list[0].args == (fake_parent_dir, 'data')
    assert tlm_parser.os.path.join.call_args_list[1].args == (fake_data_dir, 'telemetry_configs')
    assert tlm_parser.os.path.join.call_args_list[2].args == (fake_configs_dir, arg_file_name)

# mergeDicts tests
def test_tlm_json_parser_mergeDicts_when_both_args_are_empty():
    # Arrange
    arg_dict1 = {}
    arg_dict2 = {}

    # Act
    tlm_parser.mergeDicts(arg_dict1, arg_dict2)

    # Assert
    assert arg_dict1 == {}

def test_tlm_json_parser_mergeDicts_when_arg_one_is_empty_and_arg_two_is_not_empty(mocker):
    # Arrange
    arg_dict1 = {}
    dict2_len = pytest.gen.randint(1, 10) # arbitrary integer greater than 0, from 1 to 10
    arg_dict2 = {}
    [arg_dict2.update({MagicMock() : MagicMock()}) for i in range(dict2_len)]

    # Act
    tlm_parser.mergeDicts(arg_dict1, arg_dict2)

    # Assert
    assert arg_dict1 == arg_dict2

def test_tlm_json_parser_mergeDicts_when_arg_one_is_not_empty_and_arg_two_is_empty(mocker):
    # Arrange
    dict1_len = pytest.gen.randint(1, 10) # arbitrary integer greater than 0, from 1 to 10
    arg_dict1 = {}
    [arg_dict1.update({MagicMock() : MagicMock()}) for i in range(dict1_len)]
    arg_dict2 = {}

    # Act
    tlm_parser.mergeDicts(arg_dict1, arg_dict2)

    # Assert
    assert arg_dict1 == arg_dict1

def test_tlm_json_parser_mergeDicts_when_args_are_not_empty_and_have_no_shared_keys(mocker):
    # Arrange
    expected_dict1 = {}
    expected_dict2 = {}

    dict1_len = pytest.gen.randint(1, 10) # arbitrary integer greater than 0, from 1 to 10
    arg_dict1 = {}
    for i in range(dict1_len):
        key, value = MagicMock(), MagicMock()
        arg_dict1[key] = value
        expected_dict1[key] = value

    dict2_len = pytest.gen.randint(1, 10) # arbitrary integer greater than 0, from 1 to 10
    arg_dict2 = {}
    for i in range(dict2_len):
        key, value = MagicMock(), MagicMock()
        arg_dict2[key] = value
        expected_dict1[key] = value
        expected_dict2[key] = value

    # Act
    tlm_parser.mergeDicts(arg_dict1, arg_dict2)

    # Assert
    assert arg_dict1 == expected_dict1
    assert arg_dict2 == expected_dict2

# test for recursion needed