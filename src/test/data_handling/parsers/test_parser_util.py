""" Test Parser Util Functionality """
import pytest
from mock import MagicMock
import src.data_handling.parsers.parser_util as parser_util


# extract_configs tests
def test_parser_util_extract_configs_returns_empty_dicts_when_given_configFiles_is_vacant():
    # Arrange
    arg_configFilePath = MagicMock()
    arg_configFiles = []
    arg_csv = MagicMock()

    expected_result = {'subsystem_assignments' : {},
                       'test_assignments' : {},
                       'description_assignments' : {}}

    # Act
    result = parser_util.extract_configs(arg_configFilePath, arg_configFiles, arg_csv)

    # Assert
    assert result == expected_result

def test_parser_util_extract_configs_returns_expected_dicts_dict_when_configFiles_has_one_cFile(mocker):
    # Arrange
    arg_configFilePath = MagicMock()
    arg_configFiles = [MagicMock()]
    arg_csv = MagicMock()

    fake_filename = MagicMock()
    fake_subsystem_assignments = MagicMock()
    fake_tests = MagicMock()
    fake_descs = MagicMock()

    forced_return_extract_config = [fake_filename, fake_subsystem_assignments, fake_tests, fake_descs]

    mocker.patch('src.data_handling.parsers.parser_util.extract_config', return_value=forced_return_extract_config)

    expected_result = {'subsystem_assignments' : {fake_filename:fake_subsystem_assignments},
                       'test_assignments' : {fake_filename:fake_tests},
                       'description_assignments' : {fake_filename:fake_descs}}

    # Act
    result = parser_util.extract_configs(arg_configFilePath, arg_configFiles, arg_csv)

    # Assert
    assert parser_util.extract_config.call_count == 1
    assert parser_util.extract_config.call_args_list[0].args == (arg_configFilePath, arg_configFiles[0], )
    assert parser_util.extract_config.call_args_list[0].kwargs == {'csv':arg_csv}
    assert result == expected_result

def test_parser_util_extract_configs_returns_expected_dicts_dict_when_configFiles_has_many_cFile(mocker):
    # Arrange
    arg_configFilePath = MagicMock()
    arg_configFiles = [] 
    arg_csv = MagicMock()

    num_fake_cFiles = pytest.gen.randint(2, 10) # arbitrary, from 2 to 10 (0 and 1 have own tests)

    fake_filenames = [MagicMock()] * num_fake_cFiles
    fake_subsystem_assignments = [MagicMock()] * num_fake_cFiles
    fake_tests = [MagicMock()] * num_fake_cFiles
    fake_descs = [MagicMock()] * num_fake_cFiles

    forced_return_extract_configs = []
    for i in range(num_fake_cFiles):
        arg_configFiles.append(MagicMock())
        forced_return_extract_configs.append([fake_filenames[i], fake_subsystem_assignments[i], fake_tests[i], fake_descs[i]])

    mocker.patch('src.data_handling.parsers.parser_util.extract_config', side_effect=forced_return_extract_configs)

    expected_result = {}
    expected_result['subsystem_assignments'] = {fake_filenames[0]:fake_subsystem_assignments[0]}
    expected_result['test_assignments'] = {fake_filenames[0]:fake_tests[0]}
    expected_result['description_assignments'] = {fake_filenames[0]:fake_descs[0]}
    for i in range(1, num_fake_cFiles):
        expected_result['subsystem_assignments'][fake_filenames[i]] = fake_subsystem_assignments[i]
        expected_result['test_assignments'][fake_filenames[i]] = fake_tests[i]
        expected_result['description_assignments'][fake_filenames[i]] = fake_descs[i]

    # Act
    result = parser_util.extract_configs(arg_configFilePath, arg_configFiles, arg_csv)

    # Assert
    assert parser_util.extract_config.call_count == num_fake_cFiles
    for i in range(num_fake_cFiles):
        assert parser_util.extract_config.call_args_list[i].args == (arg_configFilePath, arg_configFiles[i], )
        assert parser_util.extract_config.call_args_list[i].kwargs == {'csv':arg_csv}
    assert result == expected_result

def test_parser_util_extract_configs_default_given_csv_is_False(mocker):
    # Arrange
    arg_configFilePath = MagicMock()
    arg_configFiles = [MagicMock()]

    fake_filename = MagicMock()
    fake_subsystem_assignments = MagicMock()
    fake_tests = MagicMock()
    fake_descs = MagicMock()

    forced_return_extract_config = [fake_filename, fake_subsystem_assignments, fake_tests, fake_descs]

    mocker.patch('src.data_handling.parsers.parser_util.extract_config', return_value=forced_return_extract_config)

    expected_result = {'subsystem_assignments' : {fake_filename:fake_subsystem_assignments},
                       'test_assignments' : {fake_filename:fake_tests},
                       'description_assignments' : {fake_filename:fake_descs}}

    # Act
    result = parser_util.extract_configs(arg_configFilePath, arg_configFiles)

    # Assert
    assert parser_util.extract_config.call_args_list[0].kwargs == {'csv':False}
    
# extract_config tests
def test_parser_util_extract_config_returns_tuple_of_call_to_process_file_path_and_empty_list_and_empty_list_and_empty_list_when_csv_resolves_to_True_and_dataPts_is_vacant(mocker):
    # Arrange
    arg_configFilePath = MagicMock()
    arg_configFile = []
    arg_csv = MagicMock()

    fake_data_source = MagicMock()
    fake_descriptor_file = MagicMock()
    fake_data_str = ''

    mocker.patch('src.data_handling.parsers.parser_util.process_filepath', return_value=fake_data_source)
    mocker.patch('src.data_handling.parsers.parser_util.open', return_value=fake_descriptor_file)
    mocker.patch.object(fake_descriptor_file, '.read', return_value=fake_data_str)
    mocker.patch.object(fake_descriptor_file, '.close')
    
    # Act
    result = parser_util.extract_config(arg_configFilePath, arg_configFile, arg_csv)

    # Assert
    assert parser_util.process_filepath.call_count == 1
    assert parser_util.process_filepath.call_args_list[0].args == (arg_configFile, )
    assert parser_util.process_filepath.call_args_list[0].kwargs == {'csv':arg_csv}
    assert parser_util.open.call_count == 1
    assert parser_util.open.call_args_list[0].args == (arg_configFilePath + arg_configFile, "r+")
    assert fake_descriptor_file.read.call_count == 1
    assert fake_descriptor_file.close.call_count == 1
    assert result == (fake_data_source, [], [], [])

def test_parser_util_extract_config_returns_tuple_of_call_to_process_file_path_and_empty_list_and_empty_list_and_empty_list_when_csv_resolves_to_False_and_dataPts_is_vacant(mocker):
    # Arrange
    arg_configFilePath = MagicMock()
    arg_configFile = []
    arg_csv = False if pytest.gen.randint(0, 1) == 1 else 0

    fake_data_source = MagicMock()
    fake_descriptor_file = MagicMock()
    fake_data_str = ''

    mocker.patch('src.data_handling.parsers.parser_util.process_filepath', return_value=fake_data_source)
    mocker.patch('src.data_handling.parsers.parser_util.open', return_value=fake_descriptor_file)
    mocker.patch.object(fake_descriptor_file, '.read', return_value=fake_data_str)
    mocker.patch.object(fake_descriptor_file, '.close')
    
    # Act
    result = parser_util.extract_config(arg_configFilePath, arg_configFile, arg_csv)

    # Assert
    assert parser_util.process_filepath.call_count == 1
    assert parser_util.process_filepath.call_args_list[0].args == (arg_configFile, )
    assert parser_util.process_filepath.call_args_list[0].kwargs == {'csv':arg_csv}
    assert parser_util.open.call_count == 1
    assert parser_util.open.call_args_list[0].args == (arg_configFilePath + arg_configFile, "r+")
    assert fake_descriptor_file.read.call_count == 1
    assert fake_descriptor_file.close.call_count == 1
    assert result == (fake_data_source, [], [], [])

def test_parser_util_extract_config_returns_tuple_of_call_to_process_file_path_and_3_expected_appended_lists_with_no_description_when_csv_resolves_to_True_and_dataPts_has_one_item_and_field_info_does_not_split_on_colon_and_single_test(mocker):
    # Arrange
    arg_configFilePath = MagicMock()
    arg_configFile = []
    arg_csv = MagicMock()

    fake_data_source = MagicMock()
    fake_descriptor_file = MagicMock()
    fake_descriptor = str(MagicMock()).replace(" ", "")
    fake_subsystem_assignment = MagicMock()
    fake_str_subsystem_assignment = str(fake_subsystem_assignment).replace(" ", "")
    fake_test = MagicMock()
    fake_str_test = str(fake_test).replace(" ", "")
    fake_data_str = fake_descriptor + ' ' + fake_str_subsystem_assignment + ' ' + fake_str_test

    expected_subsystem_assignments = [fake_subsystem_assignment]
    expected_mnemonic_tests = [[fake_test]]
    expected_descriptions = ["No description"]

    forced_returns_literal_eval = [fake_subsystem_assignment, fake_test]

    mocker.patch('src.data_handling.parsers.parser_util.process_filepath', return_value=fake_data_source)
    mocker.patch('src.data_handling.parsers.parser_util.open', return_value=fake_descriptor_file)
    mocker.patch.object(fake_descriptor_file, 'read', return_value=fake_data_str)
    mocker.patch.object(fake_descriptor_file, 'close')
    mocker.patch('src.data_handling.parsers.parser_util.ast.literal_eval', side_effect=forced_returns_literal_eval)
    
    # Act
    result = parser_util.extract_config(arg_configFilePath, arg_configFile, arg_csv)

    # Assert
    assert parser_util.process_filepath.call_count == 1
    assert parser_util.process_filepath.call_args_list[0].args == (arg_configFile, )
    assert parser_util.process_filepath.call_args_list[0].kwargs == {'csv':arg_csv}
    assert parser_util.open.call_count == 1
    assert parser_util.open.call_args_list[0].args == (arg_configFilePath + arg_configFile, "r+")
    assert fake_descriptor_file.read.call_count == 1
    assert fake_descriptor_file.close.call_count == 1
    assert parser_util.ast.literal_eval.call_count == 2
    assert parser_util.ast.literal_eval.call_args_list[0].args == (fake_str_subsystem_assignment, )
    assert parser_util.ast.literal_eval.call_args_list[1].args == (fake_str_test, )
    assert result == (fake_data_source, expected_subsystem_assignments, expected_mnemonic_tests, expected_descriptions)

def test_parser_util_extract_config_returns_tuple_of_call_to_process_file_path_and_3_expected_appended_lists_with_description_when_csv_resolves_to_True_and_dataPts_has_one_item_and_field_info_does_split_on_colon_and_single_test(mocker):
    # Arrange
    arg_configFilePath = MagicMock()
    arg_configFile = []
    arg_csv = MagicMock()

    fake_data_source = MagicMock()
    fake_descriptor_file = MagicMock()
    fake_descriptor = str(MagicMock()).replace(" ", "")
    fake_description = str(MagicMock())
    fake_subsystem_assignment = MagicMock()
    fake_str_subsystem_assignment = str(fake_subsystem_assignment).replace(" ", "")
    fake_test = MagicMock()
    fake_str_test = str(fake_test).replace(" ", "")
    fake_data_str = fake_descriptor + ' ' + fake_str_subsystem_assignment + ' ' + fake_str_test + ' : ' + fake_description

    expected_subsystem_assignments = [fake_subsystem_assignment]
    expected_mnemonic_tests = [[fake_test]]
    expected_descriptions = [fake_description]

    forced_returns_literal_eval = [fake_subsystem_assignment, fake_test]

    mocker.patch('src.data_handling.parsers.parser_util.process_filepath', return_value=fake_data_source)
    mocker.patch('src.data_handling.parsers.parser_util.open', return_value=fake_descriptor_file)
    mocker.patch.object(fake_descriptor_file, 'read', return_value=fake_data_str)
    mocker.patch.object(fake_descriptor_file, 'close')
    mocker.patch('src.data_handling.parsers.parser_util.ast.literal_eval', side_effect=forced_returns_literal_eval)
    
    # Act
    result = parser_util.extract_config(arg_configFilePath, arg_configFile, arg_csv)

    # Assert
    assert parser_util.process_filepath.call_count == 1
    assert parser_util.process_filepath.call_args_list[0].args == (arg_configFile, )
    assert parser_util.process_filepath.call_args_list[0].kwargs == {'csv':arg_csv}
    assert parser_util.open.call_count == 1
    assert parser_util.open.call_args_list[0].args == (arg_configFilePath + arg_configFile, "r+")
    assert fake_descriptor_file.read.call_count == 1
    assert fake_descriptor_file.close.call_count == 1
    assert parser_util.ast.literal_eval.call_count == 2
    assert parser_util.ast.literal_eval.call_args_list[0].args == (fake_str_subsystem_assignment, )
    assert parser_util.ast.literal_eval.call_args_list[1].args == (fake_str_test, )
    assert result == (fake_data_source, expected_subsystem_assignments, expected_mnemonic_tests, expected_descriptions)

def test_parser_util_extract_config_returns_tuple_of_call_to_process_file_path_and_3_expected_appended_lists_with_description_when_csv_resolves_to_True_and_dataPts_has_one_item_and_field_info_does_split_on_colon_and_multi_test(mocker):
    # Arrange
    arg_configFilePath = MagicMock()
    arg_configFile = []
    arg_csv = MagicMock()

    fake_data_source = MagicMock()
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

    expected_subsystem_assignments = [fake_subsystem_assignment]
    expected_mnemonic_tests = [[fake_test, fake_test2]]
    expected_descriptions = [fake_description]

    forced_returns_literal_eval = [fake_subsystem_assignment, fake_test, fake_test2]

    mocker.patch('src.data_handling.parsers.parser_util.process_filepath', return_value=fake_data_source)
    mocker.patch('src.data_handling.parsers.parser_util.open', return_value=fake_descriptor_file)
    mocker.patch.object(fake_descriptor_file, 'read', return_value=fake_data_str)
    mocker.patch.object(fake_descriptor_file, 'close')
    mocker.patch('src.data_handling.parsers.parser_util.ast.literal_eval', side_effect=forced_returns_literal_eval)
    
    # Act
    result = parser_util.extract_config(arg_configFilePath, arg_configFile, arg_csv)

    # Assert
    assert parser_util.process_filepath.call_count == 1
    assert parser_util.process_filepath.call_args_list[0].args == (arg_configFile, )
    assert parser_util.process_filepath.call_args_list[0].kwargs == {'csv':arg_csv}
    assert parser_util.open.call_count == 1
    assert parser_util.open.call_args_list[0].args == (arg_configFilePath + arg_configFile, "r+")
    assert fake_descriptor_file.read.call_count == 1
    assert fake_descriptor_file.close.call_count == 1
    assert parser_util.ast.literal_eval.call_count == 3
    assert parser_util.ast.literal_eval.call_args_list[0].args == (fake_str_subsystem_assignment, )
    assert parser_util.ast.literal_eval.call_args_list[1].args == (fake_str_test, )
    assert parser_util.ast.literal_eval.call_args_list[2].args == (fake_str_test2, )
    assert result == (fake_data_source, expected_subsystem_assignments, expected_mnemonic_tests, expected_descriptions)

def test_parser_util_extract_config_returns_tuple_of_call_to_process_file_path_and_empty_list_and_empty_list_and_empty_list_when_csv_resolves_to_False_and_dataPts_has_single_item(mocker):
    # Arrange
    arg_configFilePath = MagicMock()
    arg_configFile = []
    arg_csv = False if pytest.gen.randint(0, 1) == 1 else 0

    fake_data_source = MagicMock()
    fake_descriptor_file = MagicMock()
    fake_descriptor = str(MagicMock()).replace(" ", "")
    fake_subsystem_assignment = MagicMock()
    fake_str_subsystem_assignment = str(fake_subsystem_assignment).replace(" ", "")
    fake_test = MagicMock()
    fake_str_test = str(fake_test).replace(" ", "")
    fake_data_str = fake_descriptor + ' ' + fake_str_subsystem_assignment + ' ' + fake_str_test

    forced_returns_literal_eval = [fake_subsystem_assignment, fake_test]

    mocker.patch('src.data_handling.parsers.parser_util.process_filepath', return_value=fake_data_source)
    mocker.patch('src.data_handling.parsers.parser_util.open', return_value=fake_descriptor_file)
    mocker.patch.object(fake_descriptor_file, 'read', return_value=fake_data_str)
    mocker.patch.object(fake_descriptor_file, 'close')
    mocker.patch('src.data_handling.parsers.parser_util.ast.literal_eval')
        
    # Act
    result = parser_util.extract_config(arg_configFilePath, arg_configFile, arg_csv)

    # Assert
    assert parser_util.process_filepath.call_count == 1
    assert parser_util.process_filepath.call_args_list[0].args == (arg_configFile, )
    assert parser_util.process_filepath.call_args_list[0].kwargs == {'csv':arg_csv}
    assert parser_util.open.call_count == 1
    assert parser_util.open.call_args_list[0].args == (arg_configFilePath + arg_configFile, "r+")
    assert fake_descriptor_file.read.call_count == 1
    assert fake_descriptor_file.close.call_count == 1
    assert parser_util.ast.literal_eval.call_count == 0
    assert result == (fake_data_source, [], [], [])

def test_parser_util_extract_config_returns_tuple_of_call_to_process_file_path_and_3_expected_appended_lists_when_csv_resolves_to_True_and_there_are_multiple_data_points(mocker):
    # Arrange
    arg_configFilePath = MagicMock()
    arg_configFile = []
    arg_csv = MagicMock()

    fake_data_source = MagicMock()
    fake_descriptor_file = MagicMock()

    num_fake_dataPts = pytest.gen.randint(2, 10) # arbitrary, from 2 to 10 items (0 and 1 have own tests)

    fake_descriptor = str(MagicMock()).replace(" ", "")
    fake_description = str(MagicMock())
    fake_subsystem_assignment = MagicMock()
    fake_str_subsystem_assignment = str(fake_subsystem_assignment).replace(" ", "")
    fake_test = MagicMock()
    fake_str_test = str(fake_test).replace(" ", "")
    fake_data_str = ''

    forced_returns_literal_eval = []
    for i in range(num_fake_dataPts):
        fake_data_str += fake_descriptor + ' ' + fake_str_subsystem_assignment + ' ' + fake_str_test + ' : ' + fake_description + '\n'
        forced_returns_literal_eval.append(fake_subsystem_assignment)
        forced_returns_literal_eval.append(fake_test)
    # remove final newline character
    fake_data_str = fake_data_str[:-1]

    expected_subsystem_assignments = [fake_subsystem_assignment] * num_fake_dataPts
    expected_mnemonic_tests = [[fake_test]] * num_fake_dataPts
    expected_descriptions = [fake_description] * num_fake_dataPts

    mocker.patch('src.data_handling.parsers.parser_util.process_filepath', return_value=fake_data_source)
    mocker.patch('src.data_handling.parsers.parser_util.open', return_value=fake_descriptor_file)
    mocker.patch.object(fake_descriptor_file, 'read', return_value=fake_data_str)
    mocker.patch.object(fake_descriptor_file, 'close')
    mocker.patch('src.data_handling.parsers.parser_util.ast.literal_eval', side_effect=forced_returns_literal_eval)
    
    # Act
    result = parser_util.extract_config(arg_configFilePath, arg_configFile, arg_csv)

    # Assert    
    assert parser_util.process_filepath.call_count == 1
    assert parser_util.process_filepath.call_args_list[0].args == (arg_configFile, )
    assert parser_util.process_filepath.call_args_list[0].kwargs == {'csv':arg_csv}
    assert parser_util.open.call_count == 1
    assert parser_util.open.call_args_list[0].args == (arg_configFilePath + arg_configFile, "r+")
    assert fake_descriptor_file.read.call_count == 1
    assert fake_descriptor_file.close.call_count == 1
    assert parser_util.ast.literal_eval.call_count == 2 * num_fake_dataPts
    for i in range(num_fake_dataPts):
        assert parser_util.ast.literal_eval.call_args_list[2 * i].args == (fake_str_subsystem_assignment, )
        assert parser_util.ast.literal_eval.call_args_list[(2*i) + 1].args == (fake_str_test, )
    assert result == (fake_data_source, expected_subsystem_assignments, expected_mnemonic_tests, expected_descriptions)


# str2lst tests
def test_parser_util_str2lst_returns_call_to_ast_literal_eval_which_receive_given_string(mocker):
    # Arrange
    arg_string = str(MagicMock())

    expected_result = MagicMock()
    
    mocker.patch('src.data_handling.parsers.parser_util.ast.literal_eval', return_value=expected_result)

    # Act
    result = parser_util.str2lst(arg_string)

    # Assert
    assert parser_util.ast.literal_eval.call_count == 1
    assert parser_util.ast.literal_eval.call_args_list[0].args == (arg_string, )
    assert result == expected_result

def test_parser_util_str2lst_prints_message_when_ast_literal_eval_receives_given_string_but_raises_exception(mocker):
    # Arrange
    arg_string = str(MagicMock())
    
    mocker.patch('src.data_handling.parsers.parser_util.ast.literal_eval', side_effect=Exception)
    mocker.patch('src.data_handling.parsers.parser_util.print')
    
    # Act
    result = parser_util.str2lst(arg_string)

    # Assert
    assert parser_util.ast.literal_eval.call_count == 1
    assert parser_util.ast.literal_eval.call_args_list[0].args == (arg_string, )
    assert parser_util.print.call_count == 1
    assert parser_util.print.call_args_list[0].args == ("Unable to process string representation of list", )
    assert result == None

# process_filepath
def test_parser_util_process_filepath_returns_filename_from_path_with_txt_replaced_by_csv_when_given_csv_resolves_to_True_and_given_return_config_is_not_True(mocker):
    # Arrange
    fake_filename = str(MagicMock())

    arg_path = ""
    arg_return_config = False if pytest.gen.randint(0, 1) == 1 else 0
    arg_csv = True if pytest.gen.randint(0, 1) == 1 else MagicMock()

    for i in range(pytest.gen.randint(0, 10)): # arbitrary, from 0 to 10 directories in front of filename
        arg_path += str(MagicMock()) + '/'
    arg_path += fake_filename + '.txt'
    print(arg_path)
    # Act
    result = parser_util.process_filepath(arg_path, arg_return_config, arg_csv)

    # Assert
    assert result == fake_filename + '.csv'

def test_parser_util_process_filepath_returns_filename_from_path_with_txt_replaced_by__CONFIG_dot_txt_when_given_csv_resolves_to_True_and_given_return_config_is_True(mocker):
    # Arrange
    fake_filename = str(MagicMock())

    arg_path = ""
    arg_return_config = True
    arg_csv = True if pytest.gen.randint(0, 1) == 1 else MagicMock()

    for i in range(pytest.gen.randint(0, 10)): # arbitrary, from 0 to 10 directories in front of filename
        arg_path += str(MagicMock()) + '/'
    arg_path += fake_filename + '.txt'
    print(arg_path)
    # Act
    result = parser_util.process_filepath(arg_path, arg_return_config, arg_csv)

    # Assert
    assert result == fake_filename + '_CONFIG.txt'

def test_parser_util_process_filepath_returns_filename_from_path_when_given_csv_resolves_to_False_and_given_return_config_is_not_True(mocker):
    # Arrange
    fake_filename = str(MagicMock())

    arg_path = ""
    arg_return_config = False if pytest.gen.randint(0, 1) == 1 else 0
    arg_csv = False if pytest.gen.randint(0, 1) == 1 else 0

    for i in range(pytest.gen.randint(0, 10)): # arbitrary, from 0 to 10 directories in front of filename
        arg_path += str(MagicMock()) + '/'
    arg_path += fake_filename + '.txt'
    print(arg_path)
    # Act
    result = parser_util.process_filepath(arg_path, arg_return_config, arg_csv)

    # Assert
    assert result == fake_filename + '.txt'

def test_parser_util_process_filepath_returns_filename_from_path_when_given_csv_resolves_to_False_and_given_return_config_is_True(mocker):
    # Arrange
    fake_filename = str(MagicMock())

    arg_path = ""
    arg_return_config = True
    arg_csv = False if pytest.gen.randint(0, 1) == 1 else 0

    for i in range(pytest.gen.randint(0, 10)): # arbitrary, from 0 to 10 directories in front of filename
        arg_path += str(MagicMock()) + '/'
    arg_path += fake_filename + '.txt'
    print(arg_path)
    # Act
    result = parser_util.process_filepath(arg_path, arg_return_config, arg_csv)

    # Assert
    assert result == fake_filename + '_CONFIG.txt'

def test_parser_util_process_filepath_default_given_csv_is_False(mocker):
    # Arrange
    fake_filename = str(MagicMock())

    arg_path = ""
    arg_return_config = True

    for i in range(pytest.gen.randint(0, 10)): # arbitrary, from 0 to 10 directories in front of filename
        arg_path += str(MagicMock()) + '/'
    arg_path += fake_filename + '.txt'
    print(arg_path)
    # Act
    result = parser_util.process_filepath(arg_path, arg_return_config)

    # Assert
    assert result == fake_filename + '_CONFIG.txt'

def test_parser_util_process_filepath_default_given_return_config_is_False(mocker):
    # Arrange
    fake_filename = str(MagicMock())

    arg_path = ""

    for i in range(pytest.gen.randint(0, 10)): # arbitrary, from 0 to 10 directories in front of filename
        arg_path += str(MagicMock()) + '/'
    arg_path += fake_filename + '.txt'
    print(arg_path)
    # Act
    result = parser_util.process_filepath(arg_path)

    # Assert
    assert result == fake_filename + '.txt'



