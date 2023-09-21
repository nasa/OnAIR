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

import onair.data_handling.on_air_data_source as on_air_data_source
from onair.data_handling.on_air_data_source import OnAirDataSource


class FakeOnAirDataSource(OnAirDataSource):
    def process_data_file(self, data_file):
        super().process_data_file(data_file)

    def parse_meta_data_file(self, configFile, ss_breakdown):
        super().parse_meta_data_file(configFile, ss_breakdown)

    def get_next(self):
        return super().get_next()

    def has_more(self):
        return super().has_more()

    def has_data(self):
        return super().has_data()

class IncompleteOnAirDataSource(OnAirDataSource):
    pass

class BadFakeOnAirDataSource(OnAirDataSource):
    def process_data_file(self, data_file):
        return super().process_data_file(data_file)

    def parse_meta_data_file(self, configFile, ss_breakdown):
        return super().parse_meta_data_file(configFile, ss_breakdown)

    def get_next(self):
        return super().get_next()

    def has_more(self):
        return super().has_more()

    def has_data(self):
        return super().has_data()

@pytest.fixture
def setup_teardown():
    pytest.cut = FakeOnAirDataSource.__new__(FakeOnAirDataSource)
    yield 'setup_teardown'

# __init__ tests
def test_OnAirDataSource__init__sets_instance_variables_as_expected_and_calls_parse_meta_data_file_and_process_data_file(setup_teardown, mocker):
    # Arrange
    arg_rawDataFile = MagicMock()
    arg_metadataFile = MagicMock()
    arg_ss_breakdown = MagicMock()

    fake_configs = {}
    fake_configs['subsystem_assignments'] = MagicMock()
    fake_configs['test_assignments'] = MagicMock()
    fake_configs['description_assignments'] = MagicMock()
    fake_configs['data_labels'] = MagicMock()

    mocker.patch.object(pytest.cut, 'parse_meta_data_file', return_value=fake_configs)
    mocker.patch.object(pytest.cut, 'process_data_file')

    # Act
    pytest.cut.__init__(arg_rawDataFile, arg_metadataFile, arg_ss_breakdown)

    # Assert
    assert pytest.cut.raw_data_file == arg_rawDataFile
    assert pytest.cut.meta_data_file == arg_metadataFile
    assert pytest.cut.all_headers == fake_configs['data_labels']
    assert pytest.cut.sim_data == {}
    assert pytest.cut.parse_meta_data_file.call_count == 1
    assert pytest.cut.parse_meta_data_file.call_args_list[0].args == (arg_metadataFile, arg_ss_breakdown, )
    assert pytest.cut.process_data_file.call_count == 1
    assert pytest.cut.process_data_file.call_args_list[0].args == (arg_rawDataFile, )
    # assert pytest.cut.binning_configs == fake_configs
    assert pytest.cut.binning_configs['subsystem_assignments'] == fake_configs['subsystem_assignments']
    assert pytest.cut.binning_configs['test_assignments'] == fake_configs['test_assignments']
    assert pytest.cut.binning_configs['description_assignments'] == fake_configs['description_assignments']

# abstract methods tests
def test_OnAirDataSource_raises_error_because_of_unimplemented_abstract_methods():
    # Arrange - None
    # Act
    with pytest.raises(TypeError) as e_info:
        cut = OnAirDataSource.__new__(OnAirDataSource)
    
    # Assert
    assert "Can't instantiate abstract class OnAirDataSource with" in e_info.__str__()
    assert "process_data_file" in e_info.__str__()
    assert "parse_meta_data_file" in e_info.__str__()
    assert "get_next" in e_info.__str__()
    assert "has_more" in e_info.__str__()
    assert "has_data" in e_info.__str__()

# Incomplete plugin call tests
def test_OnAirDataSource_raises_error_when_an_inherited_class_is_instantiated_because_abstract_methods_are_not_implemented_by_that_class():
    # Arrange - None
    # Act
    with pytest.raises(TypeError) as e_info:
        cut = IncompleteOnAirDataSource.__new__(IncompleteOnAirDataSource)
    
    # Assert
    assert "Can't instantiate abstract class IncompleteOnAirDataSource with" in e_info.__str__()
    assert "process_data_file" in e_info.__str__()
    assert "parse_meta_data_file" in e_info.__str__()
    assert "get_next" in e_info.__str__()
    assert "has_more" in e_info.__str__()
    assert "has_data" in e_info.__str__()

def test_OnAirDataSource_raises_error_when_an_inherited_class_calls_abstract_method_process_data_file():
    # Act
    cut = BadFakeOnAirDataSource.__new__(BadFakeOnAirDataSource)

    # populate list with the functions that should raise exceptions when called.
    with pytest.raises(NotImplementedError) as e_info:
        cut.process_data_file(None)
    assert "NotImplementedError" in e_info.__str__()

def test_OnAirDataSource_raises_error_when_an_inherited_class_calls_abstract_method_parse_meta_data_file():
    # Act
    cut = BadFakeOnAirDataSource.__new__(BadFakeOnAirDataSource)

    # populate list with the functions that should raise exceptions when called.
    with pytest.raises(NotImplementedError) as e_info:
        cut.parse_meta_data_file(None, None)
    assert "NotImplementedError" in e_info.__str__()

def test_OnAirDataSource_raises_error_when_an_inherited_class_calls_abstract_method_get_next():
    # Act
    cut = BadFakeOnAirDataSource.__new__(BadFakeOnAirDataSource)

    # populate list with the functions that should raise exceptions when called.
    with pytest.raises(NotImplementedError) as e_info:
        cut.get_next()
    assert "NotImplementedError" in e_info.__str__()

def test_OnAirDataSource_raises_error_when_an_inherited_class_calls_abstract_method_has_more():
    # Act
    cut = BadFakeOnAirDataSource.__new__(BadFakeOnAirDataSource)

    # populate list with the functions that should raise exceptions when called.
    with pytest.raises(NotImplementedError) as e_info:
        cut.has_more()
    assert "NotImplementedError" in e_info.__str__()

def test_OnAirDataSource_raises_error_when_an_inherited_class_calls_abstract_method_has_data():
    # Act
    cut = BadFakeOnAirDataSource.__new__(BadFakeOnAirDataSource)

    # populate list with the functions that should raise exceptions when called.
    with pytest.raises(NotImplementedError) as e_info:
        cut.has_data()
    assert "NotImplementedError" in e_info.__str__()
