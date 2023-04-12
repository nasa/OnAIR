""" Test 42 Parser Functionality """
import pytest
from mock import MagicMock
import data_handling.parsers.on_air_parser as on_air_parser
import data_handling.parsers.forty_two_parser as forty_two_parser
from data_handling.parsers.forty_two_parser import FortyTwo


@pytest.fixture
def setup_teardown():
    pytest.cut = FortyTwo.__new__(FortyTwo)
    yield 'setup_teardown'

# tests for pre_process_data
def test_FortyTwo_pre_process_data_sets_labels_and_data_to_values_returned_from_parse_sim_data(mocker, setup_teardown):
    # Arrange
    arg_dataFiles = MagicMock()

    fake_str2lst_0 = MagicMock()
    forced_return_str2lst = [fake_str2lst_0]
    fake_labels = MagicMock()
    fake_data = MagicMock()
    forced_return_parse_sim_data = (fake_labels, fake_data)

    mocker.patch("data_handling.parsers.forty_two_parser.str2lst", return_value=forced_return_str2lst)
    mocker.patch.object(pytest.cut, "parse_sim_data", return_value=forced_return_parse_sim_data)

    # Act
    pytest.cut.pre_process_data(arg_dataFiles)

    # Assert
    assert forty_two_parser.str2lst.call_count == 1
    assert forty_two_parser.str2lst.call_args_list[0].args == (arg_dataFiles,)
    assert pytest.cut.parse_sim_data.call_count == 1
    assert pytest.cut.parse_sim_data.call_args_list[0].args == (fake_str2lst_0, )
    assert pytest.cut.all_headers == fake_labels
    assert pytest.cut.sim_data == fake_data

# tests for process_data_per_data_file
def test_FortyTwo_process_data_per_data_file_does_nothing_and_returns_None(setup_teardown):
    # Arrange
    arg_data_file = MagicMock()

    # Act
    result = pytest.cut.process_data_per_data_file(arg_data_file)

    # Assert
    assert result == None

# tests for parse sim data
def test_FortyTwo_parse_sim_data_raises_index_error_when_given_empty_data_file(mocker, setup_teardown):
    # Arrange
    arg_data_file = MagicMock()

    fake_txt_file = MagicMock()
    fake_filepath = MagicMock()
    fake_data_str = ''
    fake_headers = []

    pytest.cut.raw_data_filepath = fake_filepath

    mocker.patch('data_handling.parsers.forty_two_parser.open', return_value=fake_txt_file)
    mocker.patch.object(fake_txt_file, 'read',return_value=fake_data_str)
    mocker.patch.object(fake_txt_file, 'close')
    mocker.patch.object(pytest.cut, 'parse_headers', return_value=fake_headers)

    # Act
    with pytest.raises(IndexError) as e_info:
        pytest.cut.parse_sim_data(arg_data_file)

    # Assert
    assert e_info.match('list index out of range')
    assert forty_two_parser.open.call_count == 1
    assert fake_txt_file.read.call_count == 1
    assert fake_txt_file.close.call_count == 1
    assert pytest.cut.parse_headers.call_count == 0

def test_FortyTwo_parse_sim_data_with_only_one_data_pt(mocker, setup_teardown):
    # Arrange
    arg_data_file = MagicMock()

    fake_txt_file = MagicMock()
    fake_filepath = MagicMock()
    data_pt = str(MagicMock())
    fake_data_str = data_pt + '[EOF]\n\n'
    data_pts = [data_pt]
    
    # fake headers and frames
    fake_headers = MagicMock()
    num_frame_data = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10
    fake_frames = []
    for i in range(num_frame_data):
        fake_frames.append(MagicMock())

    pytest.cut.raw_data_filepath = fake_filepath

    mocker.patch('data_handling.parsers.forty_two_parser.open', return_value=fake_txt_file)
    mocker.patch.object(fake_txt_file, 'read',return_value=fake_data_str)
    mocker.patch.object(fake_txt_file, 'close')
    mocker.patch.object(pytest.cut, 'parse_headers', return_value=fake_headers)
    mocker.patch.object(pytest.cut, 'parse_frame', return_value=fake_frames)

    # Act
    headers_result, data_result = pytest.cut.parse_sim_data(arg_data_file)

    # Assert
    assert headers_result == {arg_data_file : fake_headers}
    assert data_result == {fake_frames[0] : {arg_data_file : fake_frames}}
    assert forty_two_parser.open.call_count == 1
    assert forty_two_parser.open.call_args_list[0].args == (fake_filepath + arg_data_file, "r+")

    assert fake_txt_file.read.call_count == 1
    assert fake_txt_file.read.call_args_list[0].args == ()
    assert fake_txt_file.close.call_count == 1
    assert fake_txt_file.close.call_args_list[0].args == ()

    assert pytest.cut.parse_headers.call_count == 1
    assert pytest.cut.parse_headers.call_args_list[0].args == (data_pts[0],)
    assert pytest.cut.parse_frame.call_count == 1
    assert pytest.cut.parse_frame.call_args_list[0].args == (data_pts[0],)

def test_FortyTwo_parse_sim_data_with_more_than_one_data_pt(mocker, setup_teardown):
    # Arrange
    arg_data_file = MagicMock()

    fake_txt_file = MagicMock()
    fake_filepath = MagicMock()
    
    # fake data pts and str
    num_data_pts = pytest.gen.randint(2, 10) # arbitrary, from 2 to 10
    data_pts = []
    fake_data_str = ''
    for i in range(num_data_pts):
        data_pt = str(MagicMock())
        fake_data_str += data_pt + '[EOF]\n\n'
        data_pts.append(data_pt)
    
    # fake headers and frames
    fake_headers = MagicMock()
    num_frame_data = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10
    fake_frames = []
    for i in range(num_frame_data):
        fake_frames.append(MagicMock())

    pytest.cut.raw_data_filepath = fake_filepath

    mocker.patch('data_handling.parsers.forty_two_parser.open', return_value=fake_txt_file)
    mocker.patch.object(fake_txt_file, 'read',return_value=fake_data_str)
    mocker.patch.object(fake_txt_file, 'close')
    mocker.patch.object(pytest.cut, 'parse_headers', return_value=fake_headers)
    mocker.patch.object(pytest.cut, 'parse_frame', return_value=fake_frames)

    # Act
    headers_result, data_result = pytest.cut.parse_sim_data(arg_data_file)

    # Assert
    assert headers_result == {arg_data_file : fake_headers}
    assert data_result == {fake_frames[0] : {arg_data_file : fake_frames}}
    assert forty_two_parser.open.call_count == 1
    assert forty_two_parser.open.call_args_list[0].args == (fake_filepath + arg_data_file, "r+")
    
    assert fake_txt_file.read.call_count == 1
    assert fake_txt_file.read.call_args_list[0].args == ()
    assert fake_txt_file.close.call_count == 1
    assert fake_txt_file.close.call_args_list[0].args == ()
    
    assert pytest.cut.parse_headers.call_count == 1
    assert pytest.cut.parse_headers.call_args_list[0].args == (data_pts[0],)
    assert pytest.cut.parse_frame.call_count == num_data_pts
    for i in range(num_data_pts):
        assert pytest.cut.parse_frame.call_args_list[i].args == (data_pts[i],)

def test_FortyTwo_parse_headers_returns_expected_value_when_given_frame_with_data(setup_teardown):
    # Arrange
    num_lines = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10
    fake_time = MagicMock()
    fake_headers = [MagicMock()] * num_lines

    expected_result = ['<MagicMock']
    arg_frame = str(fake_time)
    for fake_header in fake_headers:
        expected_result.append(str(fake_header))
        arg_frame += '\n' + str(fake_header) + ' = '
        

    # Act
    result = pytest.cut.parse_headers(arg_frame)

    # Assert
    assert result == expected_result

# tests for parse frame
def test_FortyTwo_parse_frame_raises_error_when_given_frame_with_no_data(setup_teardown):
    # Arrange
    arg_frame = ''

    # Act
    with pytest.raises(IndexError) as e_info:
        pytest.cut.parse_frame(arg_frame)

    # Assert
    assert e_info.match('list index out of range')


def test_FortyTwo_parse_frame_returns_expected_value_when_given_frame_with_data(setup_teardown):
    # Arrange
    num_lines = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10
    
    fake_time = MagicMock()
    fake_data = [MagicMock()] * num_lines

    expected_result = [str(fake_time).removeprefix('<MagicMock ')]

    arg_frame = str(fake_time)
    for fake_datum in fake_data:
        expected_result.append(str(fake_datum))
        arg_frame += '\n = ' + str(fake_datum)
    

    # Act
    result = pytest.cut.parse_frame(arg_frame)

    # Assert
    assert result == expected_result

# tests for parse headers
def test_FortyTwo_parse_header_returns_list_with_a_single_empty_string_for_frame_with_no_data(setup_teardown):
    # Arrange
    arg_frame = ''
    
    # Act
    result = pytest.cut.parse_headers(arg_frame)

    # Assert
    assert result == ['']

# tests for parse config data
def test_FortyTwo_parse_config_data_returns_expected_result_when_ss_breakdown_is_false_and_only_one_ss_assignment(mocker, setup_teardown):
    # Arrange
    arg_config_file = MagicMock()

    fake_metadata_filepath = MagicMock()
    fake_filename = MagicMock()
    fake_subsystem_assignments = [MagicMock()]
    fake_tests = MagicMock()
    fake_descs = MagicMock()

    pytest.cut.metadata_filepath = fake_metadata_filepath

    forced_return_extract_configs = { 'subsystem_assignments' : fake_subsystem_assignments,
                                        'test_assignments' : fake_tests,
                                        'description_assignments' : fake_descs}

    mocker.patch('data_handling.parsers.forty_two_parser.extract_configs', return_value=forced_return_extract_configs)
    mocker.patch('data_handling.parsers.forty_two_parser.process_filepath', return_value=fake_filename)

    expected_result = { 'subsystem_assignments' : [['MISSION']],
                        'test_assignments' : fake_tests,
                        'description_assignments' : fake_descs}

    # Act
    result = pytest.cut.parse_config_data(arg_config_file, False)
        
    # Assert
    assert result == expected_result
    assert forty_two_parser.extract_configs.call_count == 1
    assert forty_two_parser.extract_configs.call_args_list[0].args == (fake_metadata_filepath, arg_config_file)

def test_FortyTwo_parse_config_data_returns_expected_result_when_ss_breakdown_is_false_and_number_of_subsystem_assignments_greater_than_one(mocker, setup_teardown):
    # Arrange
    arg_config_file = MagicMock()

    fake_metadata_filepath = MagicMock()
    fake_filename = MagicMock()
    num_ss_assignments = pytest.gen.randint(2, 10) # arbitrary, from 2 to 10
    fake_subsystem_assignments = [MagicMock()] * num_ss_assignments
    fake_tests = MagicMock()
    fake_descs = MagicMock()

    pytest.cut.metadata_filepath = fake_metadata_filepath

    forced_return_extract_configs = { 'subsystem_assignments' : fake_subsystem_assignments,
                                        'test_assignments' : fake_tests,
                                        'description_assignments' : fake_descs}
    forced_return_process_filepath = fake_filename
    
    mocker.patch('data_handling.parsers.forty_two_parser.extract_configs', return_value=forced_return_extract_configs)
    mocker.patch('data_handling.parsers.forty_two_parser.process_filepath', return_value=forced_return_process_filepath)

    expected_result = { 'subsystem_assignments' : [['MISSION']] * num_ss_assignments,
                        'test_assignments' : fake_tests,
                        'description_assignments' : fake_descs}

    # Act
    result = pytest.cut.parse_config_data(arg_config_file, False)
        
    # Assert
    assert result == expected_result
    assert forty_two_parser.extract_configs.call_args_list[0].args == (fake_metadata_filepath, arg_config_file)

def test_FortyTwo_parse_config_data_returns_return_value_of_extract_configs_when_ss_breakdown_is_true(mocker, setup_teardown):
    # Arrange
    arg_config_file = MagicMock()

    fake_metadata_filepath = MagicMock()
    fake_filename = MagicMock()
    fake_subsystem_assignments = MagicMock()
    fake_tests = MagicMock()
    fake_descs = MagicMock()

    pytest.cut.metadata_filepath = fake_metadata_filepath

    forced_return_extract_configs = { 'subsystem_assignments' : {fake_filename:fake_subsystem_assignments},
                                        'test_assignments' : {fake_filename:fake_tests},
                                        'description_assignments' : {fake_filename:fake_descs}}
    mocker.patch('data_handling.parsers.forty_two_parser.extract_configs', return_value=forced_return_extract_configs)
    
    # Act
    result = pytest.cut.parse_config_data(arg_config_file, True)
        
    # Assert
    assert result == forced_return_extract_configs
    assert forty_two_parser.extract_configs.call_count == 1
    assert forty_two_parser.extract_configs.call_args_list[0].args == (fake_metadata_filepath, arg_config_file)

# test for get_sim_data
def test_FortyTwo_get_sim_data_returns_tuple_of_all_headers_and_sim_data_and_binning_configs_without_modifying_values(setup_teardown):
    # Arrange
    fake_all_headers = MagicMock()
    fake_sim_data = MagicMock()
    fake_binning_configs = MagicMock()
    
    pytest.cut.all_headers = fake_all_headers
    pytest.cut.sim_data = fake_sim_data
    pytest.cut.binning_configs = fake_binning_configs

    # Act
    sim_data = pytest.cut.get_sim_data()

    # Assert
    assert sim_data == (fake_all_headers, fake_sim_data, fake_binning_configs)
