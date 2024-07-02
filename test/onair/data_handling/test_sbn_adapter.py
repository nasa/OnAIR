# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

# testing packages
import pytest
from unittest.mock import MagicMock

# mock dependencies of sbn_adapter.py
import sys
sys.modules['sbn_python_client'] = MagicMock()
sys.modules['message_headers'] = MagicMock()

import onair.data_handling.sbn_adapter as sbn_adapter
from onair.data_handling.sbn_adapter import DataSource
from onair.data_handling.on_air_data_source import OnAirDataSource
from onair.data_handling.on_air_data_source import ConfigKeyError

import threading

# __init__ tests
def test_sbn_adapter_DataSource__init__sets_values_then_connects(mocker):
    # Arrange
    arg_data_file = MagicMock()
    arg_meta_file = MagicMock()
    arg_ss_breakdown = MagicMock()

    fake_new_data_lock = MagicMock()

    cut = DataSource.__new__(DataSource)

    mocker.patch.object(OnAirDataSource, '__init__', new=MagicMock())
    mocker.patch('threading.Lock', return_value=fake_new_data_lock)
    mocker.patch.object(cut, 'connect')

    # Act
    cut.__init__(arg_data_file, arg_meta_file, arg_ss_breakdown)

    # Assert
    assert OnAirDataSource.__init__.call_count == 1
    assert OnAirDataSource.__init__.call_args_list[0].args == (arg_data_file, arg_meta_file, arg_ss_breakdown)
    assert cut.new_data_lock == fake_new_data_lock
    assert cut.new_data == False
    assert cut.double_buffer_read_index == 0
    assert cut.connect.call_count == 1
    assert cut.connect.call_args_list[0].args == ()

# connect tests
# TODO !!!

# gather_field_names tests
def test_gather_field_names_returns_field_name_if_type_not_defined_in_message_headers_and_no_subfields_available(mocker):
    '''
        def gather_field_names(self, field_name, field_type):
        field_names = []
        if "message_headers" in str(field_type):
            for sub_field_name, sub_field_type in field_type._fields_:
                field_names.append(self.gather_field_names(field_name + "." + sub_field_name, sub_field_type))
        else:
            #field_names.append(field_name)
            return field_name
        return field_names
    '''
    # Arrange
    field_name = MagicMock()
    field_type = MagicMock()

    # field type was not defined in message_headers.py and has no subfields of its own
    field_type.__str__ = MagicMock()
    field_type.__str__.return_value = 'fooble'
    del field_type._fields_

    cut = DataSource.__new__(DataSource)

    # act
    result = cut.gather_field_names(field_name, field_type)

    # assert
    assert result == field_name

def test_gather_field_names_returns_nested_list_for_nested_structures(mocker):
    mock_field_name = 0
    pass

# parse_meta_data_file tests

# process_data_file tests

# get_vehicle_metadata tests

# get_next tests

# has_more tests

# mesage_listener_thread tests

# get_current_data tests