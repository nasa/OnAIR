""" Test CSV Parser Functionality """
import pytest
from mock import MagicMock
import src.data_handling.parsers.csv_parser as csv_parser
from src.data_handling.parsers.csv_parser import CSV

# CSV __init__ tests
def test_CSV__init__sets_instance_variables_as_expected_and_does_not_set_labels_and_data_and_config_when_given_dataFiles_and_configFiles_are_empty_strings_and_given_ss_breakdown_does_not_matter(mocker):
    # Arrange
    arg_rawDataFilepath = MagicMock()
    arg_metadataFilepath = MagicMock()
    arg_dataFiles = ''
    arg_configFiles = ''
    arg_ss_breakdown = MagicMock()
    
    cut = CSV.__new__(CSV)

    # Act
    cut.__init__(arg_rawDataFilepath, arg_metadataFilepath, arg_dataFiles, arg_configFiles, arg_ss_breakdown)

    # Assert
    assert cut.raw_data_file_path == arg_rawDataFilepath
    assert cut.metadata_file_path == arg_metadataFilepath
    assert cut.all_headers == ''
    assert cut.sim_data == ''
    assert cut.binning_configs == '' 

def test_CSV__init__sets_instance_variables_as_expected_and_does_not_set_labels_and_data_and_config_when_given_dataFiles_is_not_empty_string_but_configFiles_is_and_given_ss_breakdown_does_not_matter(mocker):
    # Arrange
    arg_rawDataFilepath = MagicMock()
    arg_metadataFilepath = MagicMock()
    arg_dataFiles = MagicMock()
    arg_configFiles = ''
    arg_ss_breakdown = MagicMock()
    
    cut = CSV.__new__(CSV)

    # Act
    cut.__init__(arg_rawDataFilepath, arg_metadataFilepath, arg_dataFiles, arg_configFiles, arg_ss_breakdown)

    # Assert
    assert cut.raw_data_file_path == arg_rawDataFilepath
    assert cut.metadata_file_path == arg_metadataFilepath
    assert cut.all_headers == ''
    assert cut.sim_data == ''
    assert cut.binning_configs == '' 

def test_CSV__init__sets_instance_variables_as_expected_and_does_not_set_labels_and_data_and_config_when_given_dataFiles_is_empty_string_but_configFiles_is_not_and_given_ss_breakdown_does_not_matter(mocker):
    # Arrange
    arg_rawDataFilepath = MagicMock()
    arg_metadataFilepath = MagicMock()
    arg_dataFiles = ''
    arg_configFiles = MagicMock()
    arg_ss_breakdown = MagicMock()
    
    cut = CSV.__new__(CSV)

    # Act
    cut.__init__(arg_rawDataFilepath, arg_metadataFilepath, arg_dataFiles, arg_configFiles, arg_ss_breakdown)

    # Assert
    assert cut.raw_data_file_path == arg_rawDataFilepath
    assert cut.metadata_file_path == arg_metadataFilepath
    assert cut.all_headers == ''
    assert cut.sim_data == ''
    assert cut.binning_configs == '' 

def test_CSV__init__sets_instance_variables_as_expected_and_sets_labels_and_data_and_config_and_binning_configs_dicts_are_all_empty_when_given_dataFiles_and_configFiles_are_not_empty_strings_but_dataFiles_has_no_items(mocker):
    # Arrange
    arg_rawDataFilepath = MagicMock()
    arg_metadataFilepath = MagicMock()
    arg_dataFiles = []
    arg_configFiles = MagicMock()
    arg_ss_breakdown = MagicMock()

    fake_labels = MagicMock()
    fake_data = MagicMock()
    fake_configs = {}

    expected_binning_configs = {}
    expected_binning_configs['subsystem_assignments'] = {}
    expected_binning_configs['test_assignments'] = {}
    expected_binning_configs['description_assignments'] = {}
    
    cut = CSV.__new__(CSV)

    mocker.patch('src.data_handling.parsers.csv_parser.str2lst', return_value=MagicMock())
    mocker.patch.object(cut, 'parse_csv_data', return_value=[fake_labels, fake_data])
    mocker.patch.object(cut, 'parse_config_data_CSV', return_value=fake_configs)
    mocker.patch('src.data_handling.parsers.csv_parser.ast.literal_eval', return_value=iter(arg_dataFiles))

    # Act
    cut.__init__(arg_rawDataFilepath, arg_metadataFilepath, arg_dataFiles, arg_configFiles, arg_ss_breakdown)

    # Assert
    assert cut.raw_data_file_path == arg_rawDataFilepath
    assert cut.metadata_file_path == arg_metadataFilepath
    assert cut.all_headers == fake_labels
    assert cut.sim_data == fake_data
    assert cut.binning_configs == expected_binning_configs 

def test_CSV__init__sets_instance_variables_as_expected_and_sets_labels_and_data_and_config_and_binning_configs_dicts_for_each_data_file_are_all_set_to_returned_config_values_when_given_dataFiles_and_configFiles_are_not_empty_strings(mocker):
    # Arrange
    arg_rawDataFilepath = MagicMock()
    arg_metadataFilepath = MagicMock()
    arg_dataFiles = []
    arg_configFiles = MagicMock()
    arg_ss_breakdown = MagicMock()

    fake_labels = MagicMock()
    fake_data = MagicMock()
    fake_configs = {}
    fake_configs['subsystem_assignments'] = {}
    fake_configs['test_assignments'] = {}
    fake_configs['description_assignments'] = {}

    expected_binning_configs = {}
    expected_binning_configs['subsystem_assignments'] = {}
    expected_binning_configs['test_assignments'] = {}
    expected_binning_configs['description_assignments'] = {}

    num_fake_dataFiles = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10 (0 has own test)
    for i in range(num_fake_dataFiles):
        fake_dataFile = MagicMock()
        fake_subsystem_assignments = MagicMock()
        fake_test_assignments = MagicMock()
        fake_description_assignments = MagicMock()
        fake_configs['subsystem_assignments'][fake_dataFile] = fake_subsystem_assignments
        fake_configs['test_assignments'][fake_dataFile] = fake_test_assignments
        fake_configs['description_assignments'][fake_dataFile] = fake_description_assignments
        expected_binning_configs['subsystem_assignments'][fake_dataFile] = fake_subsystem_assignments
        expected_binning_configs['test_assignments'][fake_dataFile] = fake_test_assignments
        expected_binning_configs['description_assignments'][fake_dataFile] = fake_description_assignments
        arg_dataFiles.append(fake_dataFile)
    
    cut = CSV.__new__(CSV)

    mocker.patch('src.data_handling.parsers.csv_parser.str2lst', return_value=MagicMock())
    mocker.patch.object(cut, 'parse_csv_data', return_value=[fake_labels, fake_data])
    mocker.patch.object(cut, 'parse_config_data_CSV', return_value=fake_configs)
    mocker.patch('src.data_handling.parsers.csv_parser.ast.literal_eval', return_value=iter(arg_dataFiles))

    # Act
    cut.__init__(arg_rawDataFilepath, arg_metadataFilepath, arg_dataFiles, arg_configFiles, arg_ss_breakdown)

    # Assert
    assert cut.raw_data_file_path == arg_rawDataFilepath
    assert cut.metadata_file_path == arg_metadataFilepath
    assert cut.all_headers == fake_labels
    assert cut.sim_data == fake_data
    assert cut.binning_configs == expected_binning_configs 

# CSV parse_csv_data tests
def test_CSV_parse_csv_data_returns_tuple_of_dict_with_only_given_dataFile_as_key_to_empty_list_and_empty_dict_when_parsed_dataset_from_given_dataFile_call_to_iterrows_returns_empty(mocker):
    # Arrange
    arg_dataFile = MagicMock()

    fake_raw_data_filepath = MagicMock()
    forced_return_os_path_join = MagicMock()
    fake_initial_data_set = MagicMock()
    fake_initial_data_set.loc = MagicMock()
    fake_columns_str = MagicMock()
    fake_columns = MagicMock()
    fake_columns.str = fake_columns_str
    fake_initial_data_set.columns = fake_columns
    forced_return_contains = MagicMock()
    fake_second_data_set = MagicMock()
    fake_second_data_set.columns = MagicMock()
    fake_second_data_set.columns.values = set()

    expected_result = ({arg_dataFile: []}, {})

    mocker.patch('src.data_handling.parsers.csv_parser.os.path.join', return_value=forced_return_os_path_join)
    mocker.patch('src.data_handling.parsers.csv_parser.pd.read_csv', return_value=fake_initial_data_set)
    mocker.patch.object(fake_columns_str, 'contains', return_value=forced_return_contains)
    mocker.patch.object(fake_initial_data_set, 'loc.__getitem__', return_value=fake_second_data_set)
    mocker.patch.object(fake_initial_data_set, 'iterrows', return_value=[])

    cut = CSV.__new__(CSV)
    cut.raw_data_file_path = fake_raw_data_filepath

    # Act
    result = cut.parse_csv_data(arg_dataFile)

    # Assert
    assert csv_parser.os.path.join.call_count == 1
    assert csv_parser.os.path.join.call_args_list[0].args == (fake_raw_data_filepath, arg_dataFile)
    assert csv_parser.pd.read_csv.call_count == 1
    assert csv_parser.pd.read_csv.call_args_list[0].args == (forced_return_os_path_join, )
    assert csv_parser.pd.read_csv.call_args_list[0].kwargs == {'delimiter':',', 'header':0, 'dtype':str}
    assert fake_columns_str.contains.call_count == 1
    assert fake_columns_str.contains.call_args_list[0].args == ('^Unnamed', )
    assert fake_initial_data_set.loc.__getitem__.call_args_list[0].args[0][0] == slice(None, None, None)
    assert fake_initial_data_set.loc.__getitem__.call_args_list[0].args[0][1] == ~forced_return_contains
    assert result == expected_result

def test_CSV_parse_csv_data_returns_tuple_of_dict_with_only_given_dataFile_as_key_to_empty_list_of_dataset_columns_values_and_empty_dict_when_parsed_dataset_from_given_dataFile_call_to_iterrows_returns_empty(mocker):
    # Arrange
    arg_dataFile = MagicMock()

    fake_raw_data_filepath = MagicMock()
    forced_return_os_path_join = MagicMock()
    fake_initial_data_set = MagicMock()
    fake_initial_data_set.loc = MagicMock()
    fake_columns_str = MagicMock()
    fake_columns = MagicMock()
    fake_columns.str = fake_columns_str
    fake_initial_data_set.columns = fake_columns
    forced_return_contains = MagicMock()
    fake_second_data_set = MagicMock()
    fake_second_data_set.columns = MagicMock()
    fake_second_data_set.columns.values = set()

    expected_result = ({arg_dataFile: []}, {})

    mocker.patch('src.data_handling.parsers.csv_parser.os.path.join', return_value=forced_return_os_path_join)
    mocker.patch('src.data_handling.parsers.csv_parser.pd.read_csv', return_value=fake_initial_data_set)
    mocker.patch.object(fake_columns_str, 'contains', return_value=forced_return_contains)
    mocker.patch.object(fake_initial_data_set, 'loc.__getitem__', return_value=fake_second_data_set)
    mocker.patch.object(fake_second_data_set, 'iterrows', return_value=[])

    cut = CSV.__new__(CSV)
    cut.raw_data_file_path = fake_raw_data_filepath

    # Act
    result = cut.parse_csv_data(arg_dataFile)

    # Assert
    assert csv_parser.os.path.join.call_count == 1
    assert csv_parser.os.path.join.call_args_list[0].args == (fake_raw_data_filepath, arg_dataFile)
    assert csv_parser.pd.read_csv.call_count == 1
    assert csv_parser.pd.read_csv.call_args_list[0].args == (forced_return_os_path_join, )
    assert csv_parser.pd.read_csv.call_args_list[0].kwargs == {'delimiter':',', 'header':0, 'dtype':str}
    assert fake_columns_str.contains.call_count == 1
    assert fake_columns_str.contains.call_args_list[0].args == ('^Unnamed', )
    assert fake_initial_data_set.loc.__getitem__.call_args_list[0].args[0][0] == slice(None, None, None)
    assert fake_initial_data_set.loc.__getitem__.call_args_list[0].args[0][1] == ~forced_return_contains
    assert result == expected_result

def test_CSV_parse_csv_data_returns_tuple_of_dict_with_given_dataFile_as_key_to_empty_list_of_dataset_columns_values_and_dict_of_row_values_when_parsed_dataset_from_given_dataFile_call_to_iterrows_returns_iterator(mocker):
    # Arrange
    arg_dataFile = MagicMock()

    fake_raw_data_filepath = MagicMock()
    forced_return_os_path_join = MagicMock()
    fake_initial_data_set = MagicMock()
    fake_loc = MagicMock()
    fake_initial_data_set.loc = fake_loc
    fake_columns_str = MagicMock()
    fake_columns = MagicMock()
    fake_columns.str = fake_columns_str
    fake_initial_data_set.columns = fake_columns
    forced_return_contains = MagicMock()
    fake_second_data_set = MagicMock()
    fake_second_data_set.columns = MagicMock()
    fake_second_data_set.columns.values = set()
    fake_iterrows = []
    expected_result_dict = {}
    num_fake_rows = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10 iterrows
    for i in range(num_fake_rows):
        fake_row_values = []
        for j in range(pytest.gen.randint(1,10)): # arbitrary, from 1 to 10 row values
            fake_row_values.append(pytest.gen.randint(1, 10)) # arbitrary, from 1 to 10 as a value in row
        fake_iterrows.append([i, fake_row_values])
        expected_result_dict[i] = {arg_dataFile:fake_row_values}
    forced_return_iterrows = iter(fake_iterrows)
    
    expected_result = ({arg_dataFile: []}, expected_result_dict)

    mocker.patch('src.data_handling.parsers.csv_parser.os.path.join', return_value=forced_return_os_path_join)
    mocker.patch('src.data_handling.parsers.csv_parser.pd.read_csv', return_value=fake_initial_data_set)
    mocker.patch.object(fake_columns_str, 'contains', return_value=forced_return_contains)
    mocker.patch.object(fake_loc, '__getitem__', return_value=fake_second_data_set)
    mocker.patch.object(fake_second_data_set, 'iterrows', return_value=forced_return_iterrows)

    cut = CSV.__new__(CSV)
    cut.raw_data_file_path = fake_raw_data_filepath

    # Act
    result = cut.parse_csv_data(arg_dataFile)

    # Assert
    assert csv_parser.os.path.join.call_count == 1
    assert csv_parser.os.path.join.call_args_list[0].args == (fake_raw_data_filepath, arg_dataFile)
    assert csv_parser.pd.read_csv.call_count == 1
    assert csv_parser.pd.read_csv.call_args_list[0].args == (forced_return_os_path_join, )
    assert csv_parser.pd.read_csv.call_args_list[0].kwargs == {'delimiter':',', 'header':0, 'dtype':str}
    assert fake_columns_str.contains.call_count == 1
    assert fake_columns_str.contains.call_args_list[0].args == ('^Unnamed', )
    assert result == expected_result

def test_CSV_parse_csv_data_returns_tuple_of_dict_with_given_dataFile_as_key_to_list_of_dataset_columns_values_and_dict_of_row_values_when_parsed_dataset_from_given_dataFile_call_to_iterrows_returns_iterator_and_column_names_exist_and_TIME_is_not_a_column_name(mocker):
    # Arrange
    arg_dataFile = MagicMock()

    fake_raw_data_filepath = MagicMock()
    forced_return_os_path_join = MagicMock()
    fake_initial_data_set = MagicMock()
    fake_loc = MagicMock()
    fake_initial_data_set.loc = fake_loc
    fake_columns_str = MagicMock()
    fake_columns = MagicMock()
    fake_columns.str = fake_columns_str
    fake_initial_data_set.columns = fake_columns
    forced_return_contains = MagicMock()
    fake_second_data_set = MagicMock()
    fake_second_data_set.columns = MagicMock()
    fake_second_data_set.columns.values = []
    for i in range(pytest.gen.randint(1,10)):
        fake_second_data_set.columns.values.append(str(MagicMock()))
    fake_iterrows = []
    expected_result_dict = {}
    num_fake_rows = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10 iterrows
    for i in range(num_fake_rows):
        fake_row_values = []
        for j in range(pytest.gen.randint(1,10)): # arbitrary, from 1 to 10 row values
            fake_row_values.append(pytest.gen.randint(1, 10)) # arbitrary, from 1 to 10 as a value in row
        fake_iterrows.append([i, fake_row_values])
        expected_result_dict[i] = {arg_dataFile:fake_row_values}
    forced_return_iterrows = iter(fake_iterrows)
    
    expected_result = ({arg_dataFile: fake_second_data_set.columns.values}, expected_result_dict)

    mocker.patch('src.data_handling.parsers.csv_parser.os.path.join', return_value=forced_return_os_path_join)
    mocker.patch('src.data_handling.parsers.csv_parser.pd.read_csv', return_value=fake_initial_data_set)
    mocker.patch.object(fake_columns_str, 'contains', return_value=forced_return_contains)
    mocker.patch.object(fake_loc, '__getitem__', return_value=fake_second_data_set)
    mocker.patch.object(fake_second_data_set, 'iterrows', return_value=forced_return_iterrows)

    cut = CSV.__new__(CSV)
    cut.raw_data_file_path = fake_raw_data_filepath

    # Act
    result = cut.parse_csv_data(arg_dataFile)

    # Assert
    assert csv_parser.os.path.join.call_count == 1
    assert csv_parser.os.path.join.call_args_list[0].args == (fake_raw_data_filepath, arg_dataFile)
    assert csv_parser.pd.read_csv.call_count == 1
    assert csv_parser.pd.read_csv.call_args_list[0].args == (forced_return_os_path_join, )
    assert csv_parser.pd.read_csv.call_args_list[0].kwargs == {'delimiter':',', 'header':0, 'dtype':str}
    assert fake_columns_str.contains.call_count == 1
    assert fake_columns_str.contains.call_args_list[0].args == ('^Unnamed', )
    assert result == expected_result

def test_CSV_parse_csv_data_returns_tuple_of_dict_with_given_dataFile_as_key_to_list_of_dataset_columns_values_and_dict_of_time_row_values_when_parsed_dataset_from_given_dataFile_call_to_iterrows_returns_iterator_and_column_names_exist_and_TIME_exists_as_a_column_name(mocker):
    # Arrange
    arg_dataFile = MagicMock()

    fake_raw_data_filepath = MagicMock()
    forced_return_os_path_join = MagicMock()
    fake_initial_data_set = MagicMock()
    fake_loc = MagicMock()
    fake_initial_data_set.loc = fake_loc
    fake_columns_str = MagicMock()
    fake_columns = MagicMock()
    fake_columns.str = fake_columns_str
    fake_initial_data_set.columns = fake_columns
    forced_return_contains = MagicMock()
    fake_second_data_set = MagicMock()
    fake_second_data_set.columns = MagicMock()
    fake_second_data_set.columns.values = []
    num_fake_columns = pytest.gen.randint(2,10) # arbitrary, from 2 to 10 so at least one column can be TIME
    fake_time_column_location = pytest.gen.randint(0,num_fake_columns - 1) # from 0 to 1 less than total columns
    for i in range(num_fake_columns):
        if i == fake_time_column_location:
            fake_second_data_set.columns.values.append('TIME')
        else:
            fake_second_data_set.columns.values.append(str(MagicMock()))
    fake_iterrows = []
    expected_result_dict = {}
    num_fake_rows = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10 iterrows
    counter = 0
    for i in range(num_fake_rows):
        fake_row_values = []
        for j in range(pytest.gen.randint(fake_time_column_location + 1,10)): # at least have num values up to time column location
            if j == fake_time_column_location:
                fake_row_values.append(counter) # set as counter so all times are different
                counter += 1
            else:
                fake_row_values.append(pytest.gen.randint(1, 10)) # arbitrary, from 1 to 10 as a value in row
        fake_iterrows.append([fake_row_values[fake_time_column_location], fake_row_values])
        expected_result_dict[i] = {arg_dataFile:fake_row_values}
    forced_return_iterrows = iter(fake_iterrows)
    
    expected_result = ({arg_dataFile: fake_second_data_set.columns.values}, expected_result_dict)

    mocker.patch('src.data_handling.parsers.csv_parser.os.path.join', return_value=forced_return_os_path_join)
    mocker.patch('src.data_handling.parsers.csv_parser.pd.read_csv', return_value=fake_initial_data_set)
    mocker.patch.object(fake_columns_str, 'contains', return_value=forced_return_contains)
    mocker.patch.object(fake_loc, '__getitem__', return_value=fake_second_data_set)
    mocker.patch.object(fake_second_data_set, 'iterrows', return_value=forced_return_iterrows)

    cut = CSV.__new__(CSV)
    cut.raw_data_file_path = fake_raw_data_filepath

    # Act
    result = cut.parse_csv_data(arg_dataFile)

    # Assert
    assert csv_parser.os.path.join.call_count == 1
    assert csv_parser.os.path.join.call_args_list[0].args == (fake_raw_data_filepath, arg_dataFile)
    assert csv_parser.pd.read_csv.call_count == 1
    assert csv_parser.pd.read_csv.call_args_list[0].args == (forced_return_os_path_join, )
    assert csv_parser.pd.read_csv.call_args_list[0].kwargs == {'delimiter':',', 'header':0, 'dtype':str}
    assert fake_columns_str.contains.call_count == 1
    assert fake_columns_str.contains.call_args_list[0].args == ('^Unnamed', )
    assert result == expected_result

# CSV parse_config_data_CSV tests
def test_CSV_parse_config_data_CSV_returns_call_to_extract_configs_given_metadata_filepath_and_config_File_as_single_item_list_and_kwarg_csv_set_to_True_when_given_ss_breakdown_does_not_resolve_to_False(mocker):
    # Arrange
    arg_configFile = MagicMock()
    arg_ss_breakdown = True if pytest.gen.randint(0, 1) else MagicMock()

    fake_metadata_filepath = MagicMock()

    expected_result = MagicMock()

    mocker.patch('src.data_handling.parsers.csv_parser.extract_configs', return_value=expected_result)
    mocker.patch('src.data_handling.parsers.csv_parser.len')

    cut = CSV.__new__(CSV)
    cut.metadata_file_path = fake_metadata_filepath

    # Act
    result = cut.parse_config_data_CSV(arg_configFile, arg_ss_breakdown)

    # Assert
    assert csv_parser.extract_configs.call_count == 1
    assert csv_parser.extract_configs.call_args_list[0].args == (fake_metadata_filepath, [arg_configFile])
    assert csv_parser.extract_configs.call_args_list[0].kwargs == {'csv': True}
    assert csv_parser.len.call_count == 0
    assert result == expected_result

def test_CSV_parse_config_data_CSV_returns_call_to_extract_configs_given_metadata_filepath_and_config_File_as_single_item_list_and_kwarg_csv_set_to_True_with_dict_def_of_subsystem_assigments_def_of_call_to_process_filepath_given_configFile_and_kwarg_csv_set_to_True_set_to_empty_list_when_len_of_call_value_dict_def_of_subsystem_assigments_def_of_call_to_process_filepath_given_configFile_and_kwarg_csv_set_to_True_is_0_when_given_ss_breakdown_evaluates_to_False(mocker):
    # Arrange
    arg_configFile = MagicMock()
    arg_ss_breakdown = False if pytest.gen.randint(0, 1) else 0
    
    fake_metadata_filepath = MagicMock()
    forced_return_extract_configs = {}
    forced_return_process_filepath = MagicMock()
    forced_return_len = 0
    fake_empty_processed_filepath = MagicMock()
    forced_return_extract_configs['subsystem_assignments'] = {}
    forced_return_extract_configs['subsystem_assignments'][forced_return_process_filepath] = fake_empty_processed_filepath

    expected_result = []

    mocker.patch('src.data_handling.parsers.csv_parser.extract_configs', return_value=forced_return_extract_configs)
    mocker.patch('src.data_handling.parsers.csv_parser.process_filepath', return_value=forced_return_process_filepath)
    mocker.patch('src.data_handling.parsers.csv_parser.len', return_value=forced_return_len)

    cut = CSV.__new__(CSV)
    cut.metadata_file_path = fake_metadata_filepath

    # Act
    result = cut.parse_config_data_CSV(arg_configFile, arg_ss_breakdown)

    # Assert
    assert csv_parser.extract_configs.call_count == 1
    assert csv_parser.extract_configs.call_args_list[0].args == (fake_metadata_filepath, [arg_configFile])
    assert csv_parser.extract_configs.call_args_list[0].kwargs == {'csv': True}
    assert csv_parser.process_filepath.call_count == 2
    assert csv_parser.process_filepath.call_args_list[0].args == (arg_configFile, )
    assert csv_parser.process_filepath.call_args_list[0].kwargs == {'csv': True}
    assert csv_parser.process_filepath.call_args_list[1].args == (arg_configFile, )
    assert csv_parser.process_filepath.call_args_list[1].kwargs == {'csv': True}
    assert csv_parser.len.call_count == 1
    assert csv_parser.len.call_args_list[0].args == (fake_empty_processed_filepath, )
    assert result['subsystem_assignments'][forced_return_process_filepath] == expected_result

def test_CSV_parse_config_data_CSV_returns_call_to_extract_configs_given_metadata_filepath_and_config_File_as_single_item_list_and_kwarg_csv_set_to_True_with_dict_def_subsystem_assignments_def_of_call_to_process_filepath_given_configFile_and_kwarg_csv_set_to_True_set_to_single_item_list_str_MISSION_for_each_item_when_given_ss_breakdown_evaluates_to_False(mocker):
    # Arrange
    arg_configFile = MagicMock()
    arg_ss_breakdown = False if pytest.gen.randint(0, 1) else 0
    
    fake_metadata_filepath = MagicMock()
    forced_return_extract_configs = {}
    forced_return_process_filepath = MagicMock()
    fake_processed_filepath = []
    num_fake_processed_filepaths = pytest.gen.randint(1,10) # arbitrary, from 1 to 10 (0 has own test)
    for i in range(num_fake_processed_filepaths):
        fake_processed_filepath.append(i)
    forced_return_extract_configs['subsystem_assignments'] = {}
    forced_return_extract_configs['subsystem_assignments'][forced_return_process_filepath] = fake_processed_filepath
    forced_return_len = num_fake_processed_filepaths

    expected_result = []
    for i in range(num_fake_processed_filepaths):
        expected_result.append(['MISSION'])

    mocker.patch('src.data_handling.parsers.csv_parser.extract_configs', return_value=forced_return_extract_configs)
    mocker.patch('src.data_handling.parsers.csv_parser.process_filepath', return_value=forced_return_process_filepath)
    mocker.patch('src.data_handling.parsers.csv_parser.len', return_value=forced_return_len)

    cut = CSV.__new__(CSV)
    cut.metadata_file_path = fake_metadata_filepath

    # Act
    result = cut.parse_config_data_CSV(arg_configFile, arg_ss_breakdown)

    # Assert
    assert csv_parser.extract_configs.call_count == 1
    assert csv_parser.extract_configs.call_args_list[0].args == (fake_metadata_filepath, [arg_configFile])
    assert csv_parser.extract_configs.call_args_list[0].kwargs == {'csv': True}
    assert csv_parser.process_filepath.call_count == 2
    assert csv_parser.process_filepath.call_args_list[0].args == (arg_configFile, )
    assert csv_parser.process_filepath.call_args_list[0].kwargs == {'csv': True}
    assert csv_parser.process_filepath.call_args_list[1].args == (arg_configFile, )
    assert csv_parser.process_filepath.call_args_list[1].kwargs == {'csv': True}
    assert csv_parser.len.call_count == 1
    assert csv_parser.len.call_args_list[0].args == (fake_processed_filepath, )
    assert result['subsystem_assignments'][forced_return_process_filepath] == expected_result

# CSV get_sim_data tests
def test_CSV_get_sim_data_returns_tuple_of_all_headers_and_sim_data_and_binning_configs():
    # Arrange
    fake_all_headers = MagicMock()
    fake_sim_data = MagicMock
    fake_binning_configs = MagicMock()

    expected_result = (fake_all_headers, fake_sim_data, fake_binning_configs)

    cut = CSV.__new__(CSV)
    cut.all_headers = fake_all_headers
    cut.sim_data = fake_sim_data
    cut.binning_configs = fake_binning_configs

    # Act
    result = cut.get_sim_data()

    # Assert
    assert result == expected_result