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
def test_tlm_json_parser_convertTlmToJson_calls_expected_functions_with_expected_behavior(mocker):
    # Arrange
    arg_tlm = MagicMock()
    arg_json = MagicMock()

    fake_tlm_path = MagicMock()
    fake_json_path = MagicMock()

    side_effect_list_get_config_path = [fake_tlm_path, fake_json_path]
    forced_return_parse_tlm_conf = MagicMock()
    forced_return_convert_tlm_to_json = MagicMock()

    mocker.patch('data_handling.parsers.tlm_json_parser.getConfigPath', side_effect=side_effect_list_get_config_path)
    mocker.patch('data_handling.parsers.tlm_json_parser.parseTlmConfTxt', return_value=forced_return_parse_tlm_conf)
    mocker.patch('data_handling.parsers.tlm_json_parser.convertTlmDictToJsonDict', return_value=forced_return_convert_tlm_to_json)
    mocker.patch('data_handling.parsers.tlm_json_parser.writeToJson')

    # Act
    tlm_parser.convertTlmToJson(arg_tlm, arg_json)

    # Assert
    assert tlm_parser.getConfigPath.call_count == 2
    assert tlm_parser.getConfigPath.call_args_list[0].args == (arg_tlm,)
    assert tlm_parser.getConfigPath.call_args_list[1].args == (arg_json,)
    assert tlm_parser.parseTlmConfTxt.call_count == 1
    assert tlm_parser.parseTlmConfTxt.call_args_list[0].args == (fake_tlm_path,)
    assert tlm_parser.convertTlmDictToJsonDict.call_count == 1
    assert tlm_parser.convertTlmDictToJsonDict.call_args_list[0].args == (forced_return_parse_tlm_conf,)
    assert tlm_parser.writeToJson.call_count == 1
    assert tlm_parser.writeToJson.call_args_list[0].args == (fake_json_path, forced_return_convert_tlm_to_json)

# convertTlmDictToJsonDict tests
def test_tlm_json_parser_convertTlmDictToJsonDict_returns_expected_dict_with_subsystem_NONE_and_order_empty_list_when_all_data_components_are_empty(mocker):
    # Arrange
    fake_labels = []
    fake_subsys_assigns = []
    fake_mnemonics = []
    fake_descs = []

    arg_data = [fake_labels, fake_subsys_assigns, fake_mnemonics, fake_descs]

    expected_result = {'subsystems' : {'NONE' : {}}, 'order' : []}

    mocker.patch('data_handling.parsers.tlm_json_parser.getJsonData')

    # Act
    result = tlm_parser.convertTlmDictToJsonDict(arg_data)

    # Assert
    assert tlm_parser.getJsonData.call_count == 0
    assert result == expected_result

def test_tlm_json_parser_convertTlmDictToJsonDict_raises_error_when_length_of_labels_is_not_same_as_other_data_components():
    # Arrange
    num_elems = pytest.gen.randint(0, 10) # arbitrary, from 0 to 10
    len_dif = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10
    fake_labels = [MagicMock()] * (num_elems + len_dif)
    fake_subsys_assigns = [MagicMock()] * num_elems
    fake_mnemonics = [MagicMock()] * num_elems
    fake_descs = [MagicMock()] * num_elems

    arg_data = [fake_labels, fake_subsys_assigns, fake_mnemonics, fake_descs]

    # Act
    with pytest.raises(AssertionError) as e_info:
        tlm_parser.convertTlmDictToJsonDict(arg_data)

    # Assert
    assert e_info.match('')

def test_tlm_json_parser_convertTlmDictToJsonDict_raises_error_when_length_of_subsys_assigns_is_not_same_as_other_data_components():
    # Arrange
    num_elems = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10
    len_dif = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10
    fake_labels = [MagicMock()] * num_elems
    fake_subsys_assigns = [MagicMock()] * (num_elems + len_dif)
    fake_mnemonics = [MagicMock()] * num_elems
    fake_descs = [MagicMock()] * num_elems

    arg_data = [fake_labels, fake_subsys_assigns, fake_mnemonics, fake_descs]

    # Act
    with pytest.raises(AssertionError) as e_info:
        tlm_parser.convertTlmDictToJsonDict(arg_data)

    # Assert
    assert e_info.match('')

def test_tlm_json_parser_convertTlmDictToJsonDict_raises_error_when_length_of_menmonics_is_not_same_as_other_data_components():
    # Arrange
    num_elems = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10
    len_dif = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10
    fake_labels = [MagicMock()] * num_elems
    fake_subsys_assigns = [MagicMock()] * num_elems
    fake_mnemonics = [MagicMock()] * (num_elems + len_dif)
    fake_descs = [MagicMock()] * num_elems

    arg_data = [fake_labels, fake_subsys_assigns, fake_mnemonics, fake_descs]

    # Act
    with pytest.raises(AssertionError) as e_info:
        tlm_parser.convertTlmDictToJsonDict(arg_data)

    # Assert
    assert e_info.match('')

def test_tlm_json_parser_convertTlmDictToJsonDict_raises_error_when_length_of_descriptions_is_not_same_as_other_data_components():
    # Arrange
    num_elems = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10
    len_dif = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10
    fake_labels = [MagicMock()] * num_elems
    fake_subsys_assigns = [MagicMock()] * num_elems
    fake_mnemonics = [MagicMock()] * num_elems
    fake_descs = [MagicMock()] * (num_elems + len_dif)

    arg_data = [fake_labels, fake_subsys_assigns, fake_mnemonics, fake_descs]

    # Act
    with pytest.raises(AssertionError) as e_info:
        tlm_parser.convertTlmDictToJsonDict(arg_data)

    # Assert
    assert e_info.match('')

def test_tlm_json_parser_convertTlmDictToJsonDict_returns_expected_dict_when_data_components_len_equals_one_and_subsystem_assign_contains_empty_list(mocker):
    # Arrange
    fake_labels = [MagicMock()]
    fake_subsys_assigns = [[]]
    fake_mnemonics = [MagicMock()]
    fake_descs = [MagicMock()]

    arg_data = [fake_labels, fake_subsys_assigns, fake_mnemonics, fake_descs]

    expected_result = {'subsystems' : {'NONE' : {}}, 'order' : [fake_labels[0]]}

    forced_return_get_json_data = MagicMock()
    mocker.patch('data_handling.parsers.tlm_json_parser.getJsonData', return_value=forced_return_get_json_data)
    mocker.patch('data_handling.parsers.tlm_json_parser.mergeDicts')

    # Act
    result = tlm_parser.convertTlmDictToJsonDict(arg_data)

    # Assert
    assert tlm_parser.getJsonData.call_count == 1
    assert tlm_parser.getJsonData.call_args_list[0].args == (fake_labels[0], fake_mnemonics[0], fake_descs[0])
    assert tlm_parser.mergeDicts.call_count == 1
    assert tlm_parser.mergeDicts.call_args_list[0].args == ({}, forced_return_get_json_data)
    assert result == expected_result

def test_tlm_json_parser_convertTlmDictToJsonDict_calls_mergeDicts_and_returns_expected_dict_when_len_of_all_data_components_equals_one_and_subsystem_assign_contains_one_subsys(mocker):
    # Arrange
    fake_labels = [MagicMock()]
    fake_subsys_assigns = [[MagicMock()]]
    fake_mnemonics = [MagicMock()]
    fake_descs = [MagicMock()]

    arg_data = [fake_labels, fake_subsys_assigns, fake_mnemonics, fake_descs]

    expected_result = {'subsystems' : {'NONE' : {}, fake_subsys_assigns[0][0] : {}}, 'order' : [fake_labels[0]]}

    forced_return_get_json_data = MagicMock()
    mocker.patch('data_handling.parsers.tlm_json_parser.getJsonData', return_value=forced_return_get_json_data)
    mocker.patch('data_handling.parsers.tlm_json_parser.mergeDicts')

    # Act
    result = tlm_parser.convertTlmDictToJsonDict(arg_data)

    # Assert
    assert tlm_parser.getJsonData.call_count == 1
    assert tlm_parser.getJsonData.call_args_list[0].args == (fake_labels[0], fake_mnemonics[0], fake_descs[0])
    assert tlm_parser.mergeDicts.call_count == 1
    assert tlm_parser.mergeDicts.call_args_list[0].args == ({}, forced_return_get_json_data)
    assert result == expected_result

def test_tlm_json_parser_convertTlmDictToJsonDict_calls_mergeDicts_and_returns_expected_dict_when_len_of_all_data_components_equals_one_and_subsystem_assign_contains_many_subsys(mocker):
    # Arrange
    num_elems = pytest.gen.randint(2, 10) # arbitrary, from 2 to 10
    fake_labels = [MagicMock()] * num_elems
    fake_mnemonics = [MagicMock()] * num_elems
    fake_descs = [MagicMock()] * num_elems
    expected_mergeDicts_call_count = 0
    fake_subsys_assigns = []
    for i in range(num_elems):
        num_subsys = pytest.gen.randint(2, 10) # arbitrary, from 2 to 10
        expected_mergeDicts_call_count = expected_mergeDicts_call_count + num_subsys
        fake_subsys_assigns.append([MagicMock()] * num_subsys)

    arg_data = [fake_labels, fake_subsys_assigns, fake_mnemonics, fake_descs]

    expected_result = {'subsystems' : {'NONE' : {}}, 'order' : []}
    for s_list in fake_subsys_assigns:
        for s in s_list:
            expected_result['subsystems'][s] = {}
    expected_result['order'] = fake_labels

    forced_return_get_json_data = MagicMock()
    mocker.patch('data_handling.parsers.tlm_json_parser.getJsonData', return_value=forced_return_get_json_data)
    mocker.patch('data_handling.parsers.tlm_json_parser.mergeDicts')

    # Act
    result = tlm_parser.convertTlmDictToJsonDict(arg_data)

    # Assert
    assert tlm_parser.getJsonData.call_count == num_elems
    assert tlm_parser.getJsonData.call_args_list[0].args == (fake_labels[0], fake_mnemonics[0], fake_descs[0])
    assert tlm_parser.mergeDicts.call_count == expected_mergeDicts_call_count
    assert tlm_parser.mergeDicts.call_args_list[0].args == ({}, forced_return_get_json_data)
    assert result == expected_result

def test_tlm_json_parser_convertTlmDictToJsonDict_calls_mergeDicts_and_returns_expected_dict_for_arbitrary_data_components_len_and_subsystem_assign_contains_both_empty_and_non_empty_lists(mocker):
    # Arrange
    num_non_empty_subsys_lists = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10
    num_empty_subsys_lists = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10
    expected_mergeDicts_call_count = 0
    fake_subsys_assigns = []
    for i in range(num_non_empty_subsys_lists):
        num_subsys = pytest.gen.randint(2, 10) # arbitrary, from 2 to 10
        expected_mergeDicts_call_count = expected_mergeDicts_call_count + num_subsys
        fake_subsys_assigns.append([MagicMock()] * num_subsys)
    for i in range(num_empty_subsys_lists):
        fake_subsys_assigns.append([])
    expected_mergeDicts_call_count = expected_mergeDicts_call_count + num_empty_subsys_lists

    num_elems = num_empty_subsys_lists + num_non_empty_subsys_lists
    fake_labels = [MagicMock()] * num_elems
    fake_mnemonics = [MagicMock()] * num_elems
    fake_descs = [MagicMock()] * num_elems

    arg_data = [fake_labels, fake_subsys_assigns, fake_mnemonics, fake_descs]

    expected_result = {'subsystems' : {'NONE' : {}}, 'order' : []}
    for s_list in fake_subsys_assigns:
        for s in s_list:
            expected_result['subsystems'][s] = {}
    expected_result['order'] = fake_labels

    forced_return_get_json_data = MagicMock()
    mocker.patch('data_handling.parsers.tlm_json_parser.getJsonData', return_value=forced_return_get_json_data)
    mocker.patch('data_handling.parsers.tlm_json_parser.mergeDicts')

    # Act
    result = tlm_parser.convertTlmDictToJsonDict(arg_data)

    # Assert
    assert tlm_parser.getJsonData.call_count == num_elems
    assert tlm_parser.getJsonData.call_args_list[0].args == (fake_labels[0], fake_mnemonics[0], fake_descs[0])
    assert tlm_parser.mergeDicts.call_count == expected_mergeDicts_call_count
    assert tlm_parser.mergeDicts.call_args_list[0].args == ({}, forced_return_get_json_data)
    assert result == expected_result

# getJsonData tests
def test_tlm_json_parser_getJsonData_returns_expected_data_when_label_equals_TIME():
    # Arrange
    fake_test = MagicMock()
    fake_limits = []
    
    arg_label = 'TIME'
    arg_mnemonics = [fake_test]
    arg_description = MagicMock()

    expected_result = {arg_label : {'conversion' : '', 'test' : str(fake_test), 'limits' : str(fake_limits), 'description' : str(arg_description)}}

    # Act
    result = tlm_parser.getJsonData(arg_label, arg_mnemonics, arg_description)

    # Assert
    assert result == expected_result

def test_tlm_json_parser_getJsonData_returns_expected_data_when_label_is_not_split_by_period():
    # Arrange
    fake_test = MagicMock()
    num_limits = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10
    fake_limits = [MagicMock()] * num_limits
    
    arg_label = str(MagicMock())
    arg_mnemonics = [fake_test]
    arg_mnemonics.append(fake_limits)
    arg_mnemonics = [arg_mnemonics]
    arg_description = MagicMock()

    expected_result = {arg_label : {'conversion' : '', 'test' : str(fake_test), 'limits' : str([fake_limits]), 'description' : str(arg_description)}}

    # Act
    result = tlm_parser.getJsonData(arg_label, arg_mnemonics, arg_description)

    # Assert
    assert result == expected_result

def test_tlm_json_parser_getJsonData_returns_expected_data_when_label_is_split_by_period():
    # Arrange
    fake_test = MagicMock()
    num_limits = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10
    fake_limits = [MagicMock()] * num_limits

    arg_mnemonics = [fake_test]
    arg_mnemonics.append(fake_limits)
    arg_mnemonics = [arg_mnemonics]
    arg_description = MagicMock()
    num_label_splits = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10

    label_list = [str(MagicMock())]
    arg_label = label_list[0]
    for i in range(num_label_splits):
        label = str(MagicMock())
        arg_label = label + '.' + arg_label
        label_list.append(label)

    expected_result = {'conversion' : '', 'test' : str(fake_test), 'limits' : str([fake_limits]), 'description' : str(arg_description)}
    for label in label_list:
        expected_result = {label : expected_result}

    # Act
    result = tlm_parser.getJsonData(arg_label, arg_mnemonics, arg_description)

    # Assert
    assert result == expected_result

# parseTlmConfTxt tests
def test_tlm_json_parser_parseTlmConfTxt_returns_tuple_of_empty_lists_when_dataPts_is_vacant(mocker):
    # Arrange
    arg_configFilePath = MagicMock()

    fake_descriptor_file = MagicMock()
    fake_data_str = ''

    mocker.patch('data_handling.parsers.tlm_json_parser.open', return_value=fake_descriptor_file)
    mocker.patch.object(fake_descriptor_file, 'read', return_value=fake_data_str)
    mocker.patch.object(fake_descriptor_file, 'close')
    
    # Act
    result = tlm_parser.parseTlmConfTxt(arg_configFilePath)

    # Assert
    assert tlm_parser.open.call_count == 1
    assert tlm_parser.open.call_args_list[0].args == (arg_configFilePath, 'r')
    assert fake_descriptor_file.read.call_count == 1
    assert fake_descriptor_file.close.call_count == 1
    assert result == [[], [], [], []]

def test_tlm_json_parser_parseTlmConfTxt_returns_tuple_of_4_expected_appended_lists_with_no_description_when_dataPts_has_one_item_and_field_info_does_not_split_on_colon_and_single_test(mocker):
    # Arrange
    arg_configFilePath = MagicMock()

    fake_descriptor_file = MagicMock()
    fake_descriptor = str(MagicMock()).replace(" ", "")
    fake_subsystem_assignment = MagicMock()
    fake_str_subsystem_assignment = str(fake_subsystem_assignment).replace(" ", "")
    fake_test = MagicMock()
    fake_str_test = str(fake_test).replace(" ", "")
    fake_data_str = fake_descriptor + ' ' + fake_str_subsystem_assignment + ' ' + fake_str_test

    expected_labels = [fake_descriptor]
    expected_subsystem_assignments = [fake_subsystem_assignment]
    expected_mnemonic_tests = [[fake_test]]
    expected_descriptions = ["No Description"]

    forced_returns_str2lst = [fake_subsystem_assignment, fake_test]

    mocker.patch('data_handling.parsers.tlm_json_parser.open', return_value=fake_descriptor_file)
    mocker.patch.object(fake_descriptor_file, 'read', return_value=fake_data_str)
    mocker.patch.object(fake_descriptor_file, 'close')
    mocker.patch('data_handling.parsers.tlm_json_parser.str2lst', side_effect=forced_returns_str2lst)
    
    # Act
    result = tlm_parser.parseTlmConfTxt(arg_configFilePath)

    # Assert
    assert tlm_parser.open.call_count == 1
    assert tlm_parser.open.call_args_list[0].args == (arg_configFilePath, "r")
    assert fake_descriptor_file.read.call_count == 1
    assert fake_descriptor_file.close.call_count == 1
    assert tlm_parser.str2lst.call_count == 2
    assert tlm_parser.str2lst.call_args_list[0].args == (fake_str_subsystem_assignment, )
    assert tlm_parser.str2lst.call_args_list[1].args == (fake_str_test, )
    assert result == [expected_labels, expected_subsystem_assignments, expected_mnemonic_tests, expected_descriptions]

def test_tlm_json_parser_parseTlmConfTxt_returns_tuple_of_4_expected_appended_lists_with_description_when_dataPts_has_one_item_and_field_info_does_split_on_colon_and_multi_test(mocker):
    # Arrange
    arg_configFilePath = MagicMock()

    fake_descriptor_file = MagicMock()
    fake_descriptor = str(MagicMock()).replace(" ", "")
    fake_description = str(MagicMock())
    fake_subsystem_assignment = MagicMock()
    fake_str_subsystem_assignment = str(fake_subsystem_assignment).replace(" ", "")
    fake_test = MagicMock()
    fake_str_test = str(fake_test).replace(" ", "")
    fake_test2 = MagicMock()
    fake_str_test2 = str(fake_test).replace(" ", "")
    fake_data_str = fake_descriptor + ' ' + fake_str_subsystem_assignment + ' ' + fake_str_test + ' ' + fake_str_test2 + ' : ' + fake_description

    expected_labels = [fake_descriptor]
    expected_subsystem_assignments = [fake_subsystem_assignment]
    expected_mnemonic_tests = [[fake_test, fake_test2]]
    expected_descriptions = [fake_description]

    forced_returns_literal_eval = [fake_subsystem_assignment, fake_test, fake_test2]

    mocker.patch('data_handling.parsers.tlm_json_parser.open', return_value=fake_descriptor_file)
    mocker.patch.object(fake_descriptor_file, 'read', return_value=fake_data_str)
    mocker.patch.object(fake_descriptor_file, 'close')
    mocker.patch('data_handling.parsers.tlm_json_parser.str2lst', side_effect=forced_returns_literal_eval)
    
    # Act
    result = tlm_parser.parseTlmConfTxt(arg_configFilePath)

    # Assert
    assert tlm_parser.open.call_count == 1
    assert tlm_parser.open.call_args_list[0].args == (arg_configFilePath, "r")
    assert fake_descriptor_file.read.call_count == 1
    assert fake_descriptor_file.close.call_count == 1
    assert tlm_parser.str2lst.call_count == 3
    assert tlm_parser.str2lst.call_args_list[0].args == (fake_str_subsystem_assignment, )
    assert tlm_parser.str2lst.call_args_list[1].args == (fake_str_test, )
    assert tlm_parser.str2lst.call_args_list[2].args == (fake_str_test2, )
    assert result == [expected_labels, expected_subsystem_assignments, expected_mnemonic_tests, expected_descriptions]

def test_tlm_json_parser_parseTlmConfTxt_returns_tuple_of_4_expected_appended_lists_when_there_are_multiple_data_points(mocker):
    # Arrange
    arg_configFilePath = MagicMock()

    fake_descriptor_file = MagicMock()

    num_fake_dataPts = pytest.gen.randint(2, 10) # arbitrary, from 2 to 10 items (0 and 1 have own tests)

    fake_descriptor = str(MagicMock()).replace(" ", "")
    fake_description = str(MagicMock())
    fake_subsystem_assignment = MagicMock()
    fake_str_subsystem_assignment = str(fake_subsystem_assignment).replace(" ", "")
    fake_test = MagicMock()
    fake_str_test = str(fake_test).replace(" ", "")
    fake_data_str = ''

    forced_returns_str2lst = []
    for i in range(num_fake_dataPts):
        fake_data_str += fake_descriptor + ' ' + fake_str_subsystem_assignment + ' ' + fake_str_test + ' : ' + fake_description + '\n'
        forced_returns_str2lst.append(fake_subsystem_assignment)
        forced_returns_str2lst.append(fake_test)
    # remove final newline character
    fake_data_str = fake_data_str[:-1]

    expected_labels = [fake_descriptor] * num_fake_dataPts
    expected_subsystem_assignments = [fake_subsystem_assignment] * num_fake_dataPts
    expected_mnemonic_tests = [[fake_test]] * num_fake_dataPts
    expected_descriptions = [fake_description] * num_fake_dataPts

    mocker.patch('data_handling.parsers.tlm_json_parser.open', return_value=fake_descriptor_file)
    mocker.patch.object(fake_descriptor_file, 'read', return_value=fake_data_str)
    mocker.patch.object(fake_descriptor_file, 'close')
    mocker.patch('data_handling.parsers.tlm_json_parser.str2lst', side_effect=forced_returns_str2lst)
    
    # Act
    result = tlm_parser.parseTlmConfTxt(arg_configFilePath)

    # Assert    
    assert tlm_parser.open.call_count == 1
    assert tlm_parser.open.call_args_list[0].args == (arg_configFilePath, "r")
    assert fake_descriptor_file.read.call_count == 1
    assert fake_descriptor_file.close.call_count == 1
    assert tlm_parser.str2lst.call_count == 2 * num_fake_dataPts
    for i in range(num_fake_dataPts):
        assert tlm_parser.str2lst.call_args_list[2 * i].args == (fake_str_subsystem_assignment, )
        assert tlm_parser.str2lst.call_args_list[(2*i) + 1].args == (fake_str_test, )
    assert result == [expected_labels, expected_subsystem_assignments, expected_mnemonic_tests, expected_descriptions]

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

def test_tlm_json_parser_mergeDicts_when_args_contain_shared_keys(mocker):
    # Arrange
    merged_dict = {}
    expected_dict1 = {}
    expected_dict2 = {}
    base_dicts = [{}, {}]

    for i in range(len(base_dicts)):
        dict_len = pytest.gen.randint(0, 10) # arbitrary, from 0 to 10
        for j in range(dict_len):
            key, value = str(MagicMock()), str(MagicMock())
            base_dicts[i][key] = value
            merged_dict[key] = value

    dict_len = pytest.gen.randint(0, 10) # arbitrary, from 0 to 10
    arg_dict1 = {}
    for i in range(dict_len):
        key, value = MagicMock(), MagicMock()
        arg_dict1[key] = value
        expected_dict1[key] = value

    dict_len = pytest.gen.randint(0, 10) # arbitrary, from 0 to 10
    arg_dict2 = {}
    for i in range(dict_len):
        key, value = MagicMock(), MagicMock()
        arg_dict2[key] = value
        expected_dict1[key] = value
        expected_dict2[key] = value

    num_shared_keys = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10
    shared_keys = []
    for i in range(num_shared_keys):
        key = MagicMock()
        shared_keys.append(key)
        arg_dict1[key] = base_dicts[0]
        arg_dict2[key] = base_dicts[1]
        expected_dict1[key] = merged_dict
        expected_dict2[key] = base_dicts[1]

    # Act
    tlm_parser.mergeDicts(arg_dict1, arg_dict2)

    # Assert
    assert arg_dict1 == expected_dict1
    assert arg_dict2 == expected_dict2

def test_tlm_json_parser_mergeDicts_returns_negative_one_if_arg_dict1_is_not_instance_of_dict(mocker):
    # Arrange
    arg_dict1 = MagicMock()
    arg_dict2 = {}

    # Act
    result = tlm_parser.mergeDicts(arg_dict1, arg_dict2)

    # Assert
    assert arg_dict1 == arg_dict1
    assert arg_dict2 == arg_dict2
    assert result == -1

def test_tlm_json_parser_mergeDicts_returns_negative_one_if_arg_dict2_is_not_instance_of_dict(mocker):
    # Arrange
    arg_dict1 = {}
    arg_dict2 = MagicMock()

    # Act
    result = tlm_parser.mergeDicts(arg_dict1, arg_dict2)

    # Assert
    assert arg_dict1 == arg_dict1
    assert arg_dict2 == arg_dict2
    assert result == -1