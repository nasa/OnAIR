# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

import pytest
from mock import MagicMock
import onair.src.util.file_io

# parse_associations_from_json tests

def test_file_io_parse_associations_raises_KeyError_when_loaded_data_does_not_have_keyword_children(mocker):
  # Arrange
  arg_filepath = MagicMock()

  fake_file_iterator = MagicMock()
  fake_f = MagicMock()
  fake_f.configure_mock(**{'__enter__.return_value': fake_file_iterator})

  fake_data = {}

  mocker.patch('onair.src.util.file_io.open', return_value=fake_f)
  mocker.patch('onair.src.util.file_io.json.load', return_value=fake_data)
  mocker.patch('onair.src.util.file_io.print')

  # Act
  with pytest.raises(KeyError) as e_info:
    onair.src.util.file_io.parse_associations_from_json(arg_filepath)

  # Assert
  assert str(e_info.value) == "'children'"
  assert onair.src.util.file_io.open.call_count == 1
  assert onair.src.util.file_io.json.load.call_count == 1
  assert onair.src.util.file_io.json.load.call_args_list[0].args == (fake_file_iterator,)
  assert onair.src.util.file_io.print.call_count == 0

def test_file_io_parse_associations_does_not_print_when_loaded_data_children_is_empty(mocker):
  # Arrange
  arg_filepath = MagicMock()

  fake_file_iterator = MagicMock()
  fake_f = MagicMock()
  fake_f.configure_mock(**{'__enter__.return_value': fake_file_iterator})
  fake_data = {}
  fake_data['children'] = []

  mocker.patch('onair.src.util.file_io.open', return_value=fake_f)
  mocker.patch('onair.src.util.file_io.json.load', return_value=fake_data)
  mocker.patch('onair.src.util.file_io.print')

  # Act
  onair.src.util.file_io.parse_associations_from_json(arg_filepath)

  # Assert
  assert onair.src.util.file_io.open.call_count == 1
  assert onair.src.util.file_io.json.load.call_count == 1
  assert onair.src.util.file_io.json.load.call_args_list[0].args == (fake_file_iterator,)
  assert onair.src.util.file_io.print.call_count == 0

def test_file_io_parse_associations_raises_KeyError_when_loaded_data_child_missing_name(mocker):
  # Arrange
  arg_filepath = MagicMock()

  fake_file_iterator = MagicMock()
  fake_f = MagicMock()
  fake_f.configure_mock(**{'__enter__.return_value': fake_file_iterator})
  fake_data = {}
  fake_data['children'] = [{}]

  mocker.patch('onair.src.util.file_io.open', return_value=fake_f)
  mocker.patch('onair.src.util.file_io.json.load', return_value=fake_data)
  mocker.patch('onair.src.util.file_io.print')

  # Act
  with pytest.raises(KeyError) as e_info:
    onair.src.util.file_io.parse_associations_from_json(arg_filepath)

  # Assert
  assert str(e_info.value) == "'name'"
  assert onair.src.util.file_io.open.call_count == 1
  assert onair.src.util.file_io.json.load.call_count == 1
  assert onair.src.util.file_io.json.load.call_args_list[0].args == (fake_file_iterator,)
  assert onair.src.util.file_io.print.call_count == 0

def test_file_io_parse_associations_raises_KeyError_when_loaded_data_child_missing_connections(mocker):
  # Arrange
  arg_filepath = MagicMock()

  fake_file_iterator = MagicMock()
  fake_f = MagicMock()
  fake_f.configure_mock(**{'__enter__.return_value': fake_file_iterator})
  fake_data = {}
  fake_data['children'] = [{'name':'I have a name!'}]

  mocker.patch('onair.src.util.file_io.open', return_value=fake_f)
  mocker.patch('onair.src.util.file_io.json.load', return_value=fake_data)
  mocker.patch('onair.src.util.file_io.print')

  # Act
  with pytest.raises(KeyError) as e_info:
    onair.src.util.file_io.parse_associations_from_json(arg_filepath)

  # Assert
  assert str(e_info.value) == "'connections'"
  assert onair.src.util.file_io.open.call_count == 1
  assert onair.src.util.file_io.json.load.call_count == 1
  assert onair.src.util.file_io.json.load.call_args_list[0].args == (fake_file_iterator,)
  assert onair.src.util.file_io.print.call_count == 0

def test_file_io_parse_associations_does_not_print_when_loaded_data_child_conections_are_empty(mocker):
  # Arrange
  arg_filepath = MagicMock()

  fake_file_iterator = MagicMock()
  fake_f = MagicMock()
  fake_f.configure_mock(**{'__enter__.return_value': fake_file_iterator})
  fake_data = {}
  fake_data['children'] = [{'name':'I have a name!', 'connections':[]}]

  mocker.patch('onair.src.util.file_io.open', return_value=fake_f)
  mocker.patch('onair.src.util.file_io.json.load', return_value=fake_data)
  mocker.patch('onair.src.util.file_io.print')

  # Act
  onair.src.util.file_io.parse_associations_from_json(arg_filepath)

  # Assert
  assert onair.src.util.file_io.open.call_count == 1
  assert onair.src.util.file_io.json.load.call_count == 1
  assert onair.src.util.file_io.json.load.call_args_list[0].args == (fake_file_iterator,)
  assert onair.src.util.file_io.print.call_count == 0

def test_file_io_parse_associations_raises_KeyError_when_loaded_data_child_connections_missing_target(mocker):
  # Arrange
  arg_filepath = MagicMock()

  fake_file_iterator = MagicMock()
  fake_f = MagicMock()
  fake_f.configure_mock(**{'__enter__.return_value': fake_file_iterator})
  fake_data = {}
  fake_data['children'] = [{'name':'I have a name!', 'connections':[{}]}]

  mocker.patch('onair.src.util.file_io.open', return_value=fake_f)
  mocker.patch('onair.src.util.file_io.json.load', return_value=fake_data)
  mocker.patch('onair.src.util.file_io.print')

  # Act
  with pytest.raises(KeyError) as e_info:
    onair.src.util.file_io.parse_associations_from_json(arg_filepath)

  # Assert
  assert str(e_info.value) == "'target'"
  assert onair.src.util.file_io.open.call_count == 1
  assert onair.src.util.file_io.json.load.call_count == 1
  assert onair.src.util.file_io.json.load.call_args_list[0].args == (fake_file_iterator,)
  assert onair.src.util.file_io.print.call_count == 0

def test_file_io_parse_associations_raises_KeyError_when_loaded_data_child_connections_missing_weight(mocker):
  # Arrange
  arg_filepath = MagicMock()

  fake_file_iterator = MagicMock()
  fake_f = MagicMock()
  fake_f.configure_mock(**{'__enter__.return_value': fake_file_iterator})
  fake_data = {}
  fake_data['children'] = [{'name':'I have a name!', 'connections':[{'target':'I have a target!'}]}]

  mocker.patch('onair.src.util.file_io.open', return_value=fake_f)
  mocker.patch('onair.src.util.file_io.json.load', return_value=fake_data)
  mocker.patch('onair.src.util.file_io.print')

  # Act
  with pytest.raises(KeyError) as e_info:
    onair.src.util.file_io.parse_associations_from_json(arg_filepath)

  # Assert
  assert str(e_info.value) == "'weight'"
  assert onair.src.util.file_io.open.call_count == 1
  assert onair.src.util.file_io.json.load.call_count == 1
  assert onair.src.util.file_io.json.load.call_args_list[0].args == (fake_file_iterator,)
  assert onair.src.util.file_io.print.call_count == 0

def test_file_io_parse_associations_prints_associations_in_reverse_sort_by_weight_when_data_is_properly_formed(mocker):
  # Arrange
  arg_filepath = MagicMock()

  fake_file_iterator = MagicMock()
  fake_f = MagicMock()
  fake_f.configure_mock(**{'__enter__.return_value': fake_file_iterator})
  fake_data = {}
  fake_data['children'] = []
  example_connection = {'target':'', 'weight':0}
  total_num_children = pytest.gen.randint(1, 5) # from 1 to 5
  total_num_connections = pytest.gen.randint(1, 20) # from 1 to 20
  expected_prints = []

  # fake children
  for i in range(total_num_children):
    fake_data['children'].append({'name':f"name{i}", 'connections':[]})

  # fake connections
  for i in range(total_num_connections):
    fake_target = f"target{i}"
    fake_weight = 20 - i # highest weights first
    # add to random child
    child_index = pytest.gen.randrange(0, total_num_children) # from 0 to total_num_children - 1
    fake_child = fake_data['children'][child_index]
    fake_connections = fake_child['connections']
    fake_connection = {'target':fake_target, 'weight':fake_weight}
    fake_connections.insert(pytest.gen.randint(0, len(fake_connections)), fake_connection)
    expected_prints.append(f"{fake_child['name']} --> {fake_target}, {str(fake_weight)}")

  mocker.patch('onair.src.util.file_io.open', return_value=fake_f)
  mocker.patch('onair.src.util.file_io.json.load', return_value=fake_data)
  mocker.patch('onair.src.util.file_io.print')

  # Act
  onair.src.util.file_io.parse_associations_from_json(arg_filepath)

  # Assert
  assert onair.src.util.file_io.open.call_count == 1
  assert onair.src.util.file_io.json.load.call_count == 1
  assert onair.src.util.file_io.json.load.call_args_list[0].args == (fake_file_iterator,)
  assert onair.src.util.file_io.print.call_count == total_num_connections
  for i in range(total_num_connections):
    assert onair.src.util.file_io.print.call_args_list[i].args == (expected_prints[i],)

# aggregate_results tests

def test_file_io_aggregate_results_does_nothing_then_returns_None():
  # Arrange
  expected_result = None

  # Act
  result = onair.src.util.file_io.aggregate_results()

  # Assert
  assert result == expected_result