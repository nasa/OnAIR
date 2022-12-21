""" Test Parser Util Functionality """
import pytest
from mock import MagicMock
import src.data_handling.parsers.parser_util as parser_util


# extract_configs tests
def test_extract_configs_returns_empty_dicts_when_given_configFiles_is_vacant():
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

def test_extract_configs_returns_expected_dicts_dict_when_configFiles_has_one_cFile(mocker):
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

def test_extract_configs_returns_expected_dicts_dict_when_configFiles_has_many_cFile(mocker):
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

def test_extract_configs_default_given_csv_is_False(mocker):
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

# str2lst tests
def test_str2lst_returns_call_to_ast_literal_eval_which_receive_given_string(mocker):
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

def test_str2lst_prints_message_when_ast_literal_eval_receives_given_string_but_raises_exception(mocker):
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

# process_filepath

# class TestParserUtil(unittest.TestCase):

#     def setUp(self):
#         self.run_path = os.environ['RUN_PATH']

#     def test_extract_configs(self):
#         return

#     def test_extract_config(self):
#         return

#     def test_str2lst(self):
#         return

#     def test_process_filepath(self):
#         return

# if __name__ == '__main__':
#     unittest.main()


