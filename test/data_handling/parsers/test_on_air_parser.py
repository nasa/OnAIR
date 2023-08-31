# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

""" Test OnAir Parser Functionality """
import pytest
from mock import MagicMock

import onair.data_handling.parsers.on_air_parser as on_air_parser
from onair.data_handling.parsers.on_air_parser import OnAirParser


class FakeOnAirParser(OnAirParser):
    def pre_process_data(self, dataFiles):
        super().pre_process_data(dataFiles)

    def process_data_per_data_file(self, data_file):
        super().process_data_per_data_file(data_file)

    def parse_config_data(self, configFile, ss_breakdown):
        super().parse_config_data(configFile, ss_breakdown)

class IncompleteOnAirParser(OnAirParser):
    pass

class BadFakeOnAirParser(OnAirParser):
    def pre_process_data(self, dataFiles):
        return super().pre_process_data(dataFiles)
    
    def process_data_per_data_file(self, data_file):
        return super().process_data_per_data_file(data_file)

    def parse_config_data(self, configFile, ss_breakdown):
        return super().parse_config_data(configFile, ss_breakdown)

@pytest.fixture
def setup_teardown():
    pytest.cut = FakeOnAirParser.__new__(FakeOnAirParser)
    yield 'setup_teardown'

# __init__ tests
def test_OnAirParser__init__sets_instance_variables_as_expected_and_does_not_do_configs_when_dataFiles_and_configFiles_are_empty_strings(setup_teardown):
    # Arrange
    arg_rawDataFilepath = MagicMock()
    arg_metadataFilepath = MagicMock()
    arg_dataFiles = ''
    arg_configFiles = ''
    arg_ss_breakdown = MagicMock()

    # Act
    pytest.cut.__init__(arg_rawDataFilepath, arg_metadataFilepath, arg_dataFiles, arg_configFiles, arg_ss_breakdown)

    # Assert
    assert pytest.cut.raw_data_filepath == arg_rawDataFilepath
    assert pytest.cut.metadata_filepath == arg_metadataFilepath
    assert pytest.cut.all_headers == {}
    assert pytest.cut.sim_data == {}
    assert pytest.cut.binning_configs == {}

def test_OnAirParser__init__sets_instance_variables_as_expected_and_does_not_do_configs_when_only_configFiles_is_empty_string(setup_teardown):
  # Arrange
  arg_rawDataFilepath = MagicMock()
  arg_metadataFilepath = MagicMock()
  arg_dataFiles = MagicMock()
  arg_configFiles = ''
  arg_ss_breakdown = MagicMock()

  # Act
  pytest.cut.__init__(arg_rawDataFilepath, arg_metadataFilepath, arg_dataFiles, arg_configFiles, arg_ss_breakdown)

  # Assert
  assert pytest.cut.raw_data_filepath == arg_rawDataFilepath
  assert pytest.cut.metadata_filepath == arg_metadataFilepath
  assert pytest.cut.all_headers == {}
  assert pytest.cut.sim_data == {}
  assert pytest.cut.binning_configs == {}

def test_OnAirParser__init__sets_instance_variables_as_expected_and_does_not_do_configs_when_only_dataFiles_is_empty_string(setup_teardown):
  # Arrange
  arg_rawDataFilepath = MagicMock()
  arg_metadataFilepath = MagicMock()
  arg_dataFiles = ''
  arg_configFiles = MagicMock()
  arg_ss_breakdown = MagicMock()

  # Act
  pytest.cut.__init__(arg_rawDataFilepath, arg_metadataFilepath, arg_dataFiles, arg_configFiles, arg_ss_breakdown)

  # Assert
  assert pytest.cut.raw_data_filepath == arg_rawDataFilepath
  assert pytest.cut.metadata_filepath == arg_metadataFilepath
  assert pytest.cut.all_headers == {}
  assert pytest.cut.sim_data == {}
  assert pytest.cut.binning_configs == {}

def test_OnAirParser__init__sets_instance_variables_as_expected_and_does_not_do_configs_when_dataFiles_and_configFiles_not_given_and_use_default_empty_strings(setup_teardown):
  # Arrange
  arg_rawDataFilepath = MagicMock()
  arg_metadataFilepath = MagicMock()

  # Act
  pytest.cut.__init__(arg_rawDataFilepath, arg_metadataFilepath)

  # Assert
  assert pytest.cut.raw_data_filepath == arg_rawDataFilepath
  assert pytest.cut.metadata_filepath == arg_metadataFilepath
  assert pytest.cut.all_headers == {}
  assert pytest.cut.sim_data == {}
  assert pytest.cut.binning_configs == {}

def test_OnAirParser__init__metadataFilepath_default_is_empty_string(setup_teardown):
  # Arrange
  arg_rawDataFilepath = MagicMock()

  # Act
  pytest.cut.__init__(arg_rawDataFilepath)

  # Assert
  assert pytest.cut.raw_data_filepath == arg_rawDataFilepath
  assert pytest.cut.metadata_filepath == ''
  assert pytest.cut.all_headers == {}
  assert pytest.cut.sim_data == {}
  assert pytest.cut.binning_configs == {}

def test_OnAirParser__init__rawdataFilepath_default_is_empty_string(setup_teardown):
  # Arrange - no arrangements

  # Act
  pytest.cut.__init__()

  # Assert
  assert pytest.cut.raw_data_filepath == ''
  assert pytest.cut.metadata_filepath == ''
  assert pytest.cut.all_headers == {}
  assert pytest.cut.sim_data == {}
  assert pytest.cut.binning_configs == {}

def test_OnAirParser__init__preprocesses_dataFiles_and_sets_binning_configs_when_neither_dataFiles_nor_configFiles_are_empty_string_but_str2lst_returns_empty_list(setup_teardown, mocker):
  # Arrange
  arg_rawDataFilepath = MagicMock()
  arg_metadataFilepath = MagicMock()
  arg_dataFiles = MagicMock()
  arg_configFiles = MagicMock()
  arg_ss_breakdown = MagicMock()

  fake_str2lst_first_return = [MagicMock()]

  mocker.patch.object(pytest.cut, 'pre_process_data')
  mocker.patch(on_air_parser.__name__ + '.str2lst', side_effect=[fake_str2lst_first_return, []])
  mocker.patch.object(pytest.cut, 'parse_config_data', return_value=[])

  # Act
  pytest.cut.__init__(arg_rawDataFilepath, arg_metadataFilepath, arg_dataFiles, arg_configFiles, arg_ss_breakdown)

  # Assert
  assert pytest.cut.raw_data_filepath == arg_rawDataFilepath
  assert pytest.cut.metadata_filepath == arg_metadataFilepath
  assert pytest.cut.all_headers == {}
  assert pytest.cut.sim_data == {}
  assert pytest.cut.pre_process_data.call_count == 1
  assert pytest.cut.pre_process_data.call_args_list[0].args == (arg_dataFiles, )
  assert pytest.cut.binning_configs == {'subsystem_assignments':{}, 'test_assignments':{}, 'description_assignments':{}}
  assert on_air_parser.str2lst.call_count == 2
  assert on_air_parser.str2lst.call_args_list[0].args == (arg_configFiles, )
  assert pytest.cut.parse_config_data.call_count == 1
  assert pytest.cut.parse_config_data.call_args_list[0].args == (fake_str2lst_first_return[0], arg_ss_breakdown)
  assert on_air_parser.str2lst.call_args_list[1].args == (arg_dataFiles, )

def test_OnAirParser__init__preprocesses_dataFiles_and_processes_data_per_file_setting_binning_configs_data_file_item_with_configs_when_neither_dataFiles_nor_configFiles_are_empty_string(setup_teardown, mocker):
  # Arrange
  arg_rawDataFilepath = MagicMock()
  arg_metadataFilepath = MagicMock()
  arg_dataFiles = MagicMock()
  arg_configFiles = MagicMock()
  arg_ss_breakdown = MagicMock()

  fake_str2lst_first_return = [MagicMock()]
  fake_configs = {}
  fake_configs['subsystem_assignments'] = MagicMock()
  fake_configs['test_assignments'] = MagicMock()
  fake_configs['description_assignments'] = MagicMock()
  fake_str2lst_second_return = []
  num_fake_dataFiles = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10

  for i in range(num_fake_dataFiles):
    fake_str2lst_second_return.append(MagicMock())

  mocker.patch.object(pytest.cut, 'pre_process_data')
  mocker.patch(on_air_parser.__name__ + '.str2lst', side_effect=[fake_str2lst_first_return, fake_str2lst_second_return])
  mocker.patch.object(pytest.cut, 'parse_config_data', return_value=fake_configs)
  mocker.patch.object(pytest.cut, 'process_data_per_data_file')

  # Act
  pytest.cut.__init__(arg_rawDataFilepath, arg_metadataFilepath, arg_dataFiles, arg_configFiles, arg_ss_breakdown)

  # Assert
  assert pytest.cut.raw_data_filepath == arg_rawDataFilepath
  assert pytest.cut.metadata_filepath == arg_metadataFilepath
  assert pytest.cut.all_headers == {}
  assert pytest.cut.sim_data == {}
  assert pytest.cut.pre_process_data.call_count == 1
  assert pytest.cut.pre_process_data.call_args_list[0].args == (arg_dataFiles, )
  assert on_air_parser.str2lst.call_count == 2
  assert on_air_parser.str2lst.call_args_list[0].args == (arg_configFiles, )
  assert pytest.cut.parse_config_data.call_count == 1
  assert pytest.cut.parse_config_data.call_args_list[0].args == (fake_str2lst_first_return[0], arg_ss_breakdown)
  assert on_air_parser.str2lst.call_args_list[1].args == (arg_dataFiles, )
  assert pytest.cut.process_data_per_data_file.call_count == num_fake_dataFiles
  for i in range(num_fake_dataFiles):
    assert pytest.cut.process_data_per_data_file.call_args_list[i].args == (fake_str2lst_second_return[i], )
    assert pytest.cut.binning_configs['subsystem_assignments'][fake_str2lst_second_return[i]] == fake_configs['subsystem_assignments']
    assert pytest.cut.binning_configs['test_assignments'][fake_str2lst_second_return[i]] == fake_configs['test_assignments']
    assert pytest.cut.binning_configs['description_assignments'][fake_str2lst_second_return[i]] == fake_configs['description_assignments']
 
# pre_process_data tests

# process_data_per_data_file tests

# parse_config_data tests

# abstract methods tests
def test_OnAirParser_raises_error_because_of_unimplemented_abstract_methods():
    # Arrange - None
    # Act
    with pytest.raises(TypeError) as e_info:
        cut = OnAirParser.__new__(OnAirParser)
    
    # Assert
    assert "Can't instantiate abstract class OnAirParser with" in e_info.__str__()
    assert "pre_process_data" in e_info.__str__()
    assert "process_data_per_data_file" in e_info.__str__()
    assert "parse_config_data" in e_info.__str__()

# Incomplete plugin call tests
def test_OnAirParser_raises_error_when_an_inherited_class_is_instantiated_because_abstract_methods_are_not_implemented_by_that_class():
    # Arrange - None
    # Act
    with pytest.raises(TypeError) as e_info:
        cut = IncompleteOnAirParser.__new__(IncompleteOnAirParser)
    
    # Assert
    assert "Can't instantiate abstract class IncompleteOnAirParser with" in e_info.__str__()
    assert "pre_process_data" in e_info.__str__()
    assert "process_data_per_data_file" in e_info.__str__()
    assert "parse_config_data" in e_info.__str__()

def test_OnAirParser_raises_error_when_an_inherited_class_calls_abstract_method_pre_process_data():
    # Act
    cut = BadFakeOnAirParser.__new__(BadFakeOnAirParser)

    # populate list with the functions that should raise exceptions when called.
    with pytest.raises(NotImplementedError) as e_info:
        cut.pre_process_data(None)
    assert "NotImplementedError" in e_info.__str__()

def test_OnAirParser_raises_error_when_an_inherited_class_calls_abstract_method_process_data_per_data_file():
    # Act
    cut = BadFakeOnAirParser.__new__(BadFakeOnAirParser)

    # populate list with the functions that should raise exceptions when called.
    with pytest.raises(NotImplementedError) as e_info:
        cut.process_data_per_data_file(None)
    assert "NotImplementedError" in e_info.__str__()

def test_OnAirParser_raises_error_when_an_inherited_class_calls_abstract_method_parse_config_data():
    # Act
    cut = BadFakeOnAirParser.__new__(BadFakeOnAirParser)

    # populate list with the functions that should raise exceptions when called.
    with pytest.raises(NotImplementedError) as e_info:
        cut.parse_config_data(None, None)
    assert "NotImplementedError" in e_info.__str__()
