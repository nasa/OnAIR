# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

""" Test CSV Parser Functionality """
import pytest
from mock import MagicMock

import onair.data_handling.csv_parser as csv_parser
from onair.data_handling.csv_parser import DataSource

@pytest.fixture
def setup_teardown():
    pytest.cut = DataSource.__new__(DataSource)
    yield 'setup_teardown'

# process_data_per_data_file tests
def test_CSV_process_data_file_sets_sim_data_to_parse_csv_data_return_and_frame_index_to_zero(mocker, setup_teardown):
    # Arrange
    arg_data_file = MagicMock()

    forced_return_parse_csv_data = MagicMock()

    mocker.patch.object(pytest.cut, "parse_csv_data", return_value=forced_return_parse_csv_data)

    # Act
    pytest.cut.process_data_file(arg_data_file)

    # Assert
    assert pytest.cut.sim_data == forced_return_parse_csv_data
    assert pytest.cut.frame_index == 0

# CSV parse_csv_data tests
def test_CSV_parse_csv_data_returns_empty_list_when_parsed_dataset_is_empty(mocker, setup_teardown):
    # Arrange
    arg_dataFile = MagicMock()

    fake_csv_file = MagicMock()
    fake_dataset = []
    forced_return_contains = MagicMock()
    fake_second_data_set = MagicMock()
    fake_second_data_set.columns = MagicMock()
    fake_second_data_set.columns.values = set()

    expected_result = []

    mocker.patch(csv_parser.__name__ + '.open', return_value = fake_csv_file)
    mocker.patch(csv_parser.__name__ + '.csv.reader', return_value = fake_dataset)
    mocker.patch(csv_parser.__name__ + '.floatify_input')

    # Act
    result = pytest.cut.parse_csv_data(arg_dataFile)

    # Assert
    assert csv_parser.open.call_count == 1
    assert csv_parser.open.call_args_list[0].args == (arg_dataFile, 'r')
    assert csv_parser.open.call_args_list[0].kwargs == ({'newline':''})
    assert csv_parser.csv.reader.call_count == 1
    assert csv_parser.csv.reader.call_args_list[0].args == (fake_csv_file, )
    assert csv_parser.csv.reader.call_args_list[0].kwargs == ({'delimiter':','})
    assert csv_parser.floatify_input.call_count == 0

    assert result == expected_result

def test_CSV_parse_csv_data_returns_empty_list_when_parsed_dataset_is_just_headers(mocker, setup_teardown):
    # Arrange
    arg_dataFile = MagicMock()

    fake_csv_file = MagicMock()
    fake_dataset = [['fake column header', 'another fake column header']]
    forced_return_contains = MagicMock()
    fake_second_data_set = MagicMock()
    fake_second_data_set.columns = MagicMock()
    fake_second_data_set.columns.values = set()

    expected_result = []

    mocker.patch(csv_parser.__name__ + '.open', return_value = fake_csv_file)
    mocker.patch(csv_parser.__name__ + '.csv.reader', return_value = fake_dataset)
    mocker.patch(csv_parser.__name__ + '.floatify_input')

    # Act
    result = pytest.cut.parse_csv_data(arg_dataFile)

    # Assert
    assert csv_parser.open.call_count == 1
    assert csv_parser.open.call_args_list[0].args == (arg_dataFile, 'r')
    assert csv_parser.csv.reader.call_count == 1
    assert csv_parser.csv.reader.call_args_list[0].args == (fake_csv_file, )
    assert csv_parser.csv.reader.call_args_list[0].kwargs == ({'delimiter':','})
    assert csv_parser.floatify_input.call_count == 0

    assert result == expected_result


def test_CSV_parse_csv_data_returns_list_of_row_values_when_parsed_dataset(mocker, setup_teardown):
    # Arrange
    arg_dataFile = MagicMock()

    fake_csv_file = MagicMock()
    fake_dataset = [['fake column header', 'another fake column header']]
    expected_result_list = []
    num_fake_rows = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10
    for i in range(num_fake_rows):
        fake_row_values = []
        for j in range(pytest.gen.randint(1,10)): # arbitrary, from 1 to 10 row values
            fake_row_values.append(pytest.gen.randint(1, 10)) # arbitrary, from 1 to 10 as a value in row
        fake_dataset.append([i, fake_row_values])
        expected_result_list.append(fake_row_values)

    mocker.patch(csv_parser.__name__ + '.open', return_value = fake_csv_file)
    mocker.patch(csv_parser.__name__ + '.csv.reader', return_value = fake_dataset)
    mocker.patch(csv_parser.__name__ + '.floatify_input', side_effect = expected_result_list)

    # Act
    result = pytest.cut.parse_csv_data(arg_dataFile)

    # Assert
    assert csv_parser.open.call_count == 1
    assert csv_parser.open.call_args_list[0].args == (arg_dataFile, 'r')
    assert csv_parser.csv.reader.call_count == 1
    assert csv_parser.csv.reader.call_args_list[0].args == (fake_csv_file, )
    assert csv_parser.csv.reader.call_args_list[0].kwargs == ({'delimiter':','})
    assert csv_parser.floatify_input.call_count == num_fake_rows

    assert result == expected_result_list

# CSV parse_meta_data tests
def test_CSV_parse_meta_data_file_returns_call_to_extract_meta_data_handle_ss_breakdown(mocker, setup_teardown):
    # Arrange
    arg_configFile = MagicMock()
    arg_ss_breakdown = MagicMock()

    expected_result = MagicMock()

    mocker.patch(csv_parser.__name__ + '.extract_meta_data_handle_ss_breakdown', return_value=expected_result)
    mocker.patch(csv_parser.__name__ + '.len')

    # Act
    result = pytest.cut.parse_meta_data_file(arg_configFile, arg_ss_breakdown)

    # Assert
    assert csv_parser.extract_meta_data_handle_ss_breakdown.call_count == 1
    assert csv_parser.extract_meta_data_handle_ss_breakdown.call_args_list[0].args == (arg_configFile, arg_ss_breakdown, )
    assert csv_parser.len.call_count == 0
    assert result == expected_result

# CSV get_vehicle_metadata tests
def test_CSV_get_vehicle_metadata_returns_list_of_headers_and_list_of_test_assignments(setup_teardown):
    # Arrange
    fake_all_headers = MagicMock()
    fake_test_assignments = MagicMock()
    fake_binning_configs = {}
    fake_binning_configs['test_assignments'] = fake_test_assignments

    expected_result = (fake_all_headers, fake_test_assignments)

    pytest.cut.all_headers = fake_all_headers
    pytest.cut.binning_configs = fake_binning_configs

    # Act
    result = pytest.cut.get_vehicle_metadata()

    # Assert
    assert result == expected_result

# CSV get_next test
def test_CSV_get_next_increments_index_and_returns_current_frame_of_data(setup_teardown):
    # Arrange
    fake_frame_index = 10
    fake_sim_data = []
    for i in range(fake_frame_index + 1):
        fake_sim_data.append(MagicMock())

    expected_result = fake_sim_data[fake_frame_index]

    pytest.cut.frame_index = fake_frame_index
    pytest.cut.sim_data = fake_sim_data

    # Act
    result = pytest.cut.get_next()

    # Assert
    assert result == expected_result
    assert pytest.cut.frame_index == fake_frame_index + 1

# CSV has_more test
def test_CSV_has_more_returns_true_when_index_less_than_number_of_frames(setup_teardown):
    # Arrange
    fake_frame_index = 10
    fake_sim_data = []
    for i in range(fake_frame_index + 1):
        fake_sim_data.append(MagicMock())

    expected_result = True

    pytest.cut.frame_index = 5
    pytest.cut.sim_data = fake_sim_data

    # Act
    result = pytest.cut.has_more()

    # Assert
    assert result == expected_result

def test_CSV_has_more_returns_false_when_index_equal_than_number_of_frames(setup_teardown):
    # Arrange
    fake_frame_index = 10
    fake_sim_data = []
    for i in range(fake_frame_index):
        fake_sim_data.append(MagicMock())

    expected_result = False

    pytest.cut.frame_index = fake_frame_index
    pytest.cut.sim_data = fake_sim_data

    # Act
    result = pytest.cut.has_more()

    # Assert
    assert result == expected_result
