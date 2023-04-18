""" Test Parser Util Functionality """
import pytest
from mock import MagicMock
import data_handling.parsers.parser_util as parser_util


# extract_configs tests
def test_parser_util_extract_configs_raises_error_when_given_blank_configFile():
    # Arrange
    arg_configFilePath = MagicMock()
    arg_configFile = ''
    arg_csv = MagicMock()

    # Act
    with pytest.raises(AssertionError) as e_info:
        result = parser_util.extract_configs(arg_configFilePath, arg_configFile, arg_csv)

    # Assert
    assert e_info.match('')

def test_parser_util_extract_configs_returns_expected_dicts_dict_when_configs_len_equal_to_zero(mocker):
    # Arrange
    arg_configFilePath = MagicMock()
    arg_configFile = MagicMock()
    arg_csv = MagicMock()

    fake_subsystem_assignments = MagicMock()
    fake_tests = MagicMock()
    fake_descs = MagicMock()

    forced_return_parse_tlm = {'subsystem_assignments' : fake_subsystem_assignments,
                                'test_assignments' : fake_tests,
                                'description_assignments' : fake_descs}
    forced_return_len = 0

    mocker.patch('data_handling.parsers.parser_util.parseTlmConfJson', return_value=forced_return_parse_tlm)
    mocker.patch('data_handling.parsers.parser_util.len', return_value=forced_return_len)
    mocker.patch('data_handling.parsers.parser_util.str2lst')

    # Act
    result = parser_util.extract_configs(arg_configFilePath, arg_configFile, arg_csv)

    # Assert
    assert parser_util.parseTlmConfJson.call_count == 1
    assert parser_util.parseTlmConfJson.call_args_list[0].args == (arg_configFilePath + arg_configFile, )
    assert parser_util.len.call_count == 1
    assert parser_util.len.call_args_list[0].args == (fake_subsystem_assignments, )
    assert parser_util.str2lst.call_count == 0
    assert result == forced_return_parse_tlm

def test_parser_util_extract_configs_returns_expected_dicts_dict_when_configs_len_equal_to_one(mocker):
    # Arrange
    arg_configFilePath = MagicMock()
    arg_configFile = MagicMock()
    arg_csv = MagicMock()

    fake_subsystem_assignments = [MagicMock()]
    fake_test_assign = [MagicMock()]
    fake_tests = [fake_test_assign]
    fake_descs = [MagicMock()]

    forced_return_parse_tlm = {'subsystem_assignments' : fake_subsystem_assignments,
                                'test_assignments' : fake_tests,
                                'description_assignments' : fake_descs}
    forced_return_str2lst = MagicMock()

    mocker.patch('data_handling.parsers.parser_util.parseTlmConfJson', return_value=forced_return_parse_tlm)
    mocker.patch('data_handling.parsers.parser_util.str2lst', return_value=forced_return_str2lst)

    expected_ss_assigns = [[fake_ss_assign] for fake_ss_assign in fake_subsystem_assignments]
    expected_result = {}
    expected_result['subsystem_assignments'] = expected_ss_assigns
    expected_result['test_assignments'] = [[forced_return_str2lst]]
    expected_result['description_assignments'] = fake_descs.copy()

    # Act
    result = parser_util.extract_configs(arg_configFilePath, arg_configFile, arg_csv)

    # Assert
    assert parser_util.parseTlmConfJson.call_count == 1
    assert parser_util.parseTlmConfJson.call_args_list[0].args == (arg_configFilePath + arg_configFile, )
    assert parser_util.str2lst.call_count == 1
    assert parser_util.str2lst.call_args_list[0].args == (fake_test_assign[0], )
    assert result == forced_return_parse_tlm

def test_parser_util_extract_configs_returns_expected_dicts_dict_when_len_configs_greater_than_one(mocker):
    # Arrange
    arg_configFilePath = MagicMock()
    arg_configFile = MagicMock()
    arg_csv = MagicMock()

    len_configs = pytest.gen.randint(2, 10) # arbitrary, from 2 to 10 (0 and 1 have own tests)
    fake_subsystem_assignments = [MagicMock()] * len_configs
    fake_test_assign = [MagicMock()]
    fake_tests = [fake_test_assign] * len_configs
    fake_descs = [MagicMock()] * len_configs

    forced_return_parse_tlm = {'subsystem_assignments' : fake_subsystem_assignments,
                                'test_assignments' : fake_tests,
                                'description_assignments' : fake_descs}
    forced_return_str2lst = MagicMock()

    mocker.patch('data_handling.parsers.parser_util.parseTlmConfJson', return_value=forced_return_parse_tlm)
    mocker.patch('data_handling.parsers.parser_util.str2lst', return_value=forced_return_str2lst)

    expected_ss_assigns = [[fake_ss_assign] for fake_ss_assign in fake_subsystem_assignments]
    expected_result = {}
    expected_result['subsystem_assignments'] = expected_ss_assigns
    expected_result['test_assignments'] = [[forced_return_str2lst]] * len_configs
    expected_result['description_assignments'] = fake_descs.copy()

    # Act
    result = parser_util.extract_configs(arg_configFilePath, arg_configFile, arg_csv)

    # Assert
    assert parser_util.parseTlmConfJson.call_count == 1
    assert parser_util.parseTlmConfJson.call_args_list[0].args == (arg_configFilePath + arg_configFile, )
    assert parser_util.str2lst.call_count == len_configs
    for i in range(len_configs):
        assert parser_util.str2lst.call_args_list[i].args == (fake_test_assign[0], )
    assert result == expected_result
def test_parser_util_extract_configs_returns_expected_dicts_dict_when_len_configs_greater_than_one_and_NOOPs_contained_in_test_assigns(mocker):
    # Arrange
    arg_configFilePath = MagicMock()
    arg_configFile = MagicMock() 
    arg_csv = MagicMock()

    len_configs = pytest.gen.randint(2, 10) # arbitrary, from 2 to 10 (0 and 1 have own tests)
    num_noops = pytest.gen.randint(2, 10)
    len_configs = len_configs + num_noops
    fake_subsystem_assignments = [MagicMock()] * len_configs
    fake_test_assign = [MagicMock()]
    noop_test_assign = [['NOOP']]
    fake_tests = [[fake_test_assign]] * (len_configs - num_noops) + noop_test_assign * num_noops
    fake_descs = [MagicMock()] * len_configs
    
    forced_return_parse_tlm = {'subsystem_assignments' : fake_subsystem_assignments,
                                'test_assignments' : fake_tests,
                                'description_assignments' : fake_descs}
    forced_return_str2lst = MagicMock()

    mocker.patch('data_handling.parsers.parser_util.parseTlmConfJson', return_value=forced_return_parse_tlm)
    mocker.patch('data_handling.parsers.parser_util.str2lst', return_value=forced_return_str2lst)

    expected_ss_assigns = [[fake_ss_assign] for fake_ss_assign in fake_subsystem_assignments]
    expected_result = {}
    expected_result['subsystem_assignments'] = expected_ss_assigns
    expected_result['test_assignments'] = [[forced_return_str2lst]] * (len_configs - num_noops) + [noop_test_assign] * num_noops
    expected_result['description_assignments'] = fake_descs.copy()

    # Act
    result = parser_util.extract_configs(arg_configFilePath, arg_configFile, arg_csv)

    # Assert
    assert parser_util.parseTlmConfJson.call_count == 1
    assert parser_util.parseTlmConfJson.call_args_list[0].args == (arg_configFilePath + arg_configFile, )
    assert parser_util.str2lst.call_count == len_configs - num_noops
    for i in range(len_configs - num_noops):
        assert parser_util.str2lst.call_args_list[i].args == (fake_test_assign, )
    assert result == expected_result

def test_parser_util_extract_configs_returns_expected_dicts_dict_when_len_configs_greater_than_one_and_len_test_assigns_greater_than_one(mocker):
    # Arrange
    arg_configFilePath = MagicMock()
    arg_configFile = MagicMock()
    arg_csv = MagicMock()

    len_configs = pytest.gen.randint(2, 10) # arbitrary, from 2 to 10 (0 and 1 have own tests)
    fake_subsystem_assignments = [MagicMock()] * len_configs
    fake_tests= []
    fake_descs = [MagicMock()] * len_configs
    for i in range(len_configs):
        len_test_assigns = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10
        fake_test_assigns = [MagicMock()] * len_test_assigns
        fake_tests.append(fake_test_assigns)

    forced_return_parse_tlm = {'subsystem_assignments' : fake_subsystem_assignments,
                                'test_assignments' : fake_tests,
                                'description_assignments' : fake_descs}
    forced_return_str2lst = [MagicMock()]

    mocker.patch('data_handling.parsers.parser_util.parseTlmConfJson', return_value=forced_return_parse_tlm)
    mocker.patch('data_handling.parsers.parser_util.str2lst', return_value=forced_return_str2lst)

    expected_ss_assigns = [[fake_ss_assign] for fake_ss_assign in fake_subsystem_assignments]
    expected_result = {}
    expected_result['subsystem_assignments'] = expected_ss_assigns
    expected_result['test_assignments'] = fake_tests.copy()
    expected_result['description_assignments'] = fake_descs.copy()

    expected_str2lst_args = []

    for i in range(len_configs):
        if len(fake_tests[i]) > 1:
            expected_result['test_assignments'][i] = [[fake_tests[i][0]] + forced_return_str2lst]
            expected_str2lst_args.append(fake_tests[i][1])
        else:
            expected_result['test_assignments'][i] = [forced_return_str2lst]
            expected_str2lst_args.append(fake_tests[i][0])

    # Act
    result = parser_util.extract_configs(arg_configFilePath, arg_configFile, arg_csv)

    # Assert
    assert parser_util.parseTlmConfJson.call_count == 1
    assert parser_util.parseTlmConfJson.call_args_list[0].args == (arg_configFilePath + arg_configFile, )
    assert parser_util.str2lst.call_count == len_configs
    for i in range(len_configs):
        assert parser_util.str2lst.call_args_list[i].args == (expected_str2lst_args[i], )
    assert result == expected_result

def test_parser_util_extract_configs_returns_expected_dicts_dict_when_configFiles_has_many_cFile_and_len_configs_less_than_two(mocker):
    # Arrange
    arg_configFilePath = MagicMock()
    arg_configFile = MagicMock()
    arg_csv = MagicMock()

    len_configs = pytest.gen.randint(0, 1) # arbitrary, from 0 to 1
    fake_subsystem_assignments = [MagicMock()] * len_configs
    fake_tests = []
    fake_tests_copy = []
    fake_descs = [MagicMock()] * len_configs

    for i in range(len_configs):
        fake_test_assign = [MagicMock()]
        fake_tests.append(fake_test_assign)
        fake_tests_copy.append(fake_test_assign)

    forced_return_parse_tlm = {'subsystem_assignments' : fake_subsystem_assignments,
                                'test_assignments' : fake_tests,
                                'description_assignments' : fake_descs}
    forced_return_str2lst = [MagicMock()]

    mocker.patch('data_handling.parsers.parser_util.parseTlmConfJson', return_value=forced_return_parse_tlm)
    mocker.patch('data_handling.parsers.parser_util.str2lst', return_value=forced_return_str2lst)

    expected_ss_assigns = [[fake_ss_assign] for fake_ss_assign in fake_subsystem_assignments]
    expected_result = {}
    expected_result['subsystem_assignments'] = expected_ss_assigns
    expected_result['test_assignments'] = [[forced_return_str2lst]] * len_configs
    expected_result['description_assignments'] = fake_descs.copy()

    # Act
    result = parser_util.extract_configs(arg_configFilePath, arg_configFile, arg_csv)

    # Assert
    assert parser_util.parseTlmConfJson.call_count == 1
    assert parser_util.parseTlmConfJson.call_args_list[0].args == (arg_configFilePath + arg_configFile, )
    assert parser_util.str2lst.call_count == len_configs
    for i in range(len_configs):
        assert parser_util.str2lst.call_args_list[i].args == (fake_tests_copy[i][0], )
    assert result == expected_result

def test_parser_util_extract_configs_returns_expected_dicts_dict_when_configFiles_has_many_cFile_and_len_configs_greater_than_one(mocker):
    # Arrange
    arg_configFilePath = MagicMock()
    arg_configFile = MagicMock()
    arg_csv = MagicMock()

    len_configs = pytest.gen.randint(2, 10) # arbitrary, from 2 to 10 (0 and 1 have own tests)
    fake_subsystem_assignments = [MagicMock()] * len_configs
    fake_tests = []
    fake_tests_copy = []
    fake_descs = [MagicMock()] * len_configs

    for i in range(len_configs):
        fake_test_assign = [MagicMock()]
        fake_tests.append(fake_test_assign)
        fake_tests_copy.append(fake_test_assign)

    forced_return_parse_tlm = {'subsystem_assignments' : fake_subsystem_assignments,
                                'test_assignments' : fake_tests,
                                'description_assignments' : fake_descs}
    forced_return_str2lst = [MagicMock()]

    mocker.patch('data_handling.parsers.parser_util.parseTlmConfJson', return_value=forced_return_parse_tlm)
    mocker.patch('data_handling.parsers.parser_util.str2lst', return_value=forced_return_str2lst)

    expected_ss_assigns = [[fake_ss_assign] for fake_ss_assign in fake_subsystem_assignments]
    expected_result = {}
    expected_result['subsystem_assignments'] = expected_ss_assigns
    expected_result['test_assignments'] = [[forced_return_str2lst]] * len_configs
    expected_result['description_assignments'] = fake_descs.copy()

    # Act
    result = parser_util.extract_configs(arg_configFilePath, arg_configFile, arg_csv)

    # Assert
    assert parser_util.parseTlmConfJson.call_count == 1
    assert parser_util.parseTlmConfJson.call_args_list[0].args == (arg_configFilePath + arg_configFile, )
    assert parser_util.str2lst.call_count == len_configs
    for i in range(len_configs):
        assert parser_util.str2lst.call_args_list[i].args == (fake_tests_copy[i][0], )
    assert result == expected_result

def test_parser_util_extract_configs_returns_expected_dicts_dict_when_configFiles_has_many_cFile_and_len_configs_greater_than_one_and_subsystem_NONE_exists(mocker):
    # Arrange
    arg_configFilePath = MagicMock()
    arg_configFile = MagicMock()
    arg_csv = MagicMock()

    len_configs = pytest.gen.randint(2, 10) # arbitrary, from 2 to 10 (0 and 1 have own tests)
    fake_subsystem_assignments = []
    fake_tests = []
    fake_tests_copy = []
    fake_descs = []

    expected_subsystem_assignments = []
    for i in range(len_configs):
        if pytest.gen.randint(0, 1) == 1:
            fake_ss_assign = MagicMock()
            fake_subsystem_assignments.append(fake_ss_assign)
            expected_subsystem_assignments.append([fake_ss_assign])
        else:
            fake_subsystem_assignments.append('NONE')
            expected_subsystem_assignments.append([])
        fake_test_assign = [MagicMock()]
        fake_tests.append(fake_test_assign)
        fake_tests_copy.append(fake_test_assign)
        fake_descs.append(MagicMock())

    rand_index = pytest.gen.randint(0, len_configs-1) # arbitrary index in fake_subsystem_assignments
    fake_subsystem_assignments[rand_index] = 'NONE'
    expected_subsystem_assignments[rand_index] = []

    forced_return_parse_tlm = {'subsystem_assignments' : fake_subsystem_assignments,
                                'test_assignments' : fake_tests,
                                'description_assignments' : fake_descs}
    forced_return_str2lst = [MagicMock()]

    mocker.patch('data_handling.parsers.parser_util.parseTlmConfJson', return_value=forced_return_parse_tlm)
    mocker.patch('data_handling.parsers.parser_util.str2lst', return_value=forced_return_str2lst)

    expected_result = {}
    expected_result['subsystem_assignments'] = expected_subsystem_assignments
    expected_result['test_assignments'] = [[forced_return_str2lst]] * len_configs
    expected_result['description_assignments'] = fake_descs.copy()

    # Act
    result = parser_util.extract_configs(arg_configFilePath, arg_configFile, arg_csv)

    # Assert
    assert parser_util.parseTlmConfJson.call_count == 1
    assert parser_util.parseTlmConfJson.call_args_list[0].args == (arg_configFilePath + arg_configFile, )
    assert parser_util.str2lst.call_count == len_configs
    for i in range(len_configs):
        assert parser_util.str2lst.call_args_list[i].args == (fake_tests_copy[i][0], )
    assert result == expected_result

# process_filepath
def test_parser_util_process_filepath_returns_filename_from_path_with_txt_replaced_by_csv_when_given_csv_resolves_to_True_and_given_return_config_is_not_True(mocker):
    # Arrange
    fake_filename = str(MagicMock())
    fake_os_sep = pytest.gen.choice(chr(47) + chr(92)) # representative separators, 47 = '/', 92 = '\'

    arg_path = ""
    arg_return_config = False if pytest.gen.randint(0, 1) == 1 else 0
    arg_csv = True if pytest.gen.randint(0, 1) == 1 else MagicMock()

    for i in range(pytest.gen.randint(0, 10)): # arbitrary, from 0 to 10 directories in front of filename
        arg_path += str(MagicMock()) + fake_os_sep
    arg_path += fake_filename + '.txt'

    mocker.patch('data_handling.parsers.parser_util.os.sep', fake_os_sep)

    # Act
    result = parser_util.process_filepath(arg_path, arg_return_config, arg_csv)

    # Assert
    assert result == fake_filename + '.csv'

def test_parser_util_process_filepath_returns_filename_from_path_with_txt_replaced_by__CONFIG_dot_txt_when_given_csv_resolves_to_True_and_given_return_config_is_True(mocker):
    # Arrange
    fake_filename = str(MagicMock())
    fake_os_sep = pytest.gen.choice(chr(47) + chr(92)) # representative separators, 47 = '/', 92 = '\'

    arg_path = ""
    arg_return_config = True
    arg_csv = True if pytest.gen.randint(0, 1) == 1 else MagicMock()

    for i in range(pytest.gen.randint(0, 10)): # arbitrary, from 0 to 10 directories in front of filename
        arg_path += str(MagicMock()) + fake_os_sep
    arg_path += fake_filename + '.txt'

    mocker.patch('data_handling.parsers.parser_util.os.sep', fake_os_sep)

    # Act
    result = parser_util.process_filepath(arg_path, arg_return_config, arg_csv)

    # Assert
    assert result == fake_filename + '_CONFIG.txt'

def test_parser_util_process_filepath_returns_filename_from_path_when_given_csv_resolves_to_False_and_given_return_config_is_not_True(mocker):
    # Arrange
    fake_filename = str(MagicMock())
    fake_os_sep = pytest.gen.choice(chr(47) + chr(92)) # representative separators, 47 = '/', 92 = '\'

    arg_path = ""
    arg_return_config = False if pytest.gen.randint(0, 1) == 1 else 0
    arg_csv = False if pytest.gen.randint(0, 1) == 1 else 0

    for i in range(pytest.gen.randint(0, 10)): # arbitrary, from 0 to 10 directories in front of filename
        arg_path += str(MagicMock()) + fake_os_sep
    arg_path += fake_filename + '.txt'

    mocker.patch('data_handling.parsers.parser_util.os.sep', fake_os_sep)

    # Act
    result = parser_util.process_filepath(arg_path, arg_return_config, arg_csv)

    # Assert
    assert result == fake_filename + '.txt'

def test_parser_util_process_filepath_returns_filename_from_path_when_given_csv_resolves_to_False_and_given_return_config_is_True(mocker):
    # Arrange
    fake_filename = str(MagicMock())
    fake_os_sep = pytest.gen.choice(chr(47) + chr(92)) # representative separators, 47 = '/', 92 = '\'

    arg_path = ""
    arg_return_config = True
    arg_csv = False if pytest.gen.randint(0, 1) == 1 else 0

    for i in range(pytest.gen.randint(0, 10)): # arbitrary, from 0 to 10 directories in front of filename
        arg_path += str(MagicMock()) + fake_os_sep
    arg_path += fake_filename + '.txt'

    mocker.patch('data_handling.parsers.parser_util.os.sep', fake_os_sep)

    # Act
    result = parser_util.process_filepath(arg_path, arg_return_config, arg_csv)

    # Assert
    assert result == fake_filename + '_CONFIG.txt'

def test_parser_util_process_filepath_default_given_csv_is_False(mocker):
    # Arrange
    fake_filename = str(MagicMock())
    fake_os_sep = pytest.gen.choice(chr(47) + chr(92)) # representative separators, 47 = '/', 92 = '\'

    arg_path = ""
    arg_return_config = True

    for i in range(pytest.gen.randint(0, 10)): # arbitrary, from 0 to 10 directories in front of filename
        arg_path += str(MagicMock()) + fake_os_sep
    arg_path += fake_filename + '.txt'

    mocker.patch('data_handling.parsers.parser_util.os.sep', fake_os_sep)

    # Act
    result = parser_util.process_filepath(arg_path, arg_return_config)

    # Assert
    assert result == fake_filename + '_CONFIG.txt'

def test_parser_util_process_filepath_default_given_return_config_is_False(mocker):
    # Arrange
    fake_filename = str(MagicMock())
    fake_os_sep = pytest.gen.choice(chr(47) + chr(92)) # representative separators, 47 = '/', 92 = '\'

    arg_path = ""

    for i in range(pytest.gen.randint(0, 10)): # arbitrary, from 0 to 10 directories in front of filename
        arg_path += str(MagicMock()) + fake_os_sep
    arg_path += fake_filename + '.txt'

    mocker.patch('data_handling.parsers.parser_util.os.sep', fake_os_sep)

    # Act
    result = parser_util.process_filepath(arg_path)

    # Assert
    assert result == fake_filename + '.txt'



