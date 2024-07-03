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
from unittest.mock import MagicMock, PropertyMock

# mock dependencies of sbn_adapter.py
import sys
sys.modules['sbn_python_client'] = MagicMock()
sys.modules['message_headers'] = MagicMock()

import onair.data_handling.sbn_adapter as sbn_adapter
from onair.data_handling.sbn_adapter import DataSource
from onair.data_handling.on_air_data_source import OnAirDataSource
from onair.data_handling.on_air_data_source import ConfigKeyError

import threading
import datetime
import copy
import json

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
def test_sbn_adapter_DataSource_gather_field_names_returns_field_name_if_type_not_defined_in_message_headers_and_no_subfields_available(mocker):
    # Arrange
    cut = DataSource.__new__(DataSource)

    field_name = MagicMock()
    field_type = MagicMock()

    # field type was not defined in message_headers.py and has no subfields of its own
    field_type.__str__ = MagicMock()
    field_type.__str__.return_value = 'fooble'
    del field_type._fields_

    # Act
    result = cut.gather_field_names(field_name, field_type)

    # Assert
    # TODO return type depends on bugfix(?) in
    assert result == [field_name]

def test_sbn_adapter_Data_Source_gather_field_names_returns_nested_list_for_nested_structure(mocker):
    # Arrange
    cut = DataSource.__new__(DataSource)

    # parent field has two child fields.
    # The first child field has a grandchild field
    parent_field_name = "parent_field"
    parent_field_type = MagicMock()
    child1_field_name = "child1_field"
    child1_field_type = MagicMock()
    child2_field_name = "child2_field"
    child2_field_type = MagicMock()
    gchild_field_name = "gchild_field"
    gchild_field_type = MagicMock()

    gchild_field_type.__str__ = MagicMock()
    gchild_field_type.__str__.return_value = "message_headers.mock_data_type"
    del gchild_field_type._fields_

    child2_field_type.__str__ = MagicMock()
    child2_field_type.__str__.return_value = "message_headers.mock_data_type"
    del child2_field_type._fields_

    child1_field_type.__str__ = MagicMock()
    child1_field_type.__str__.return_value = "message_headers.mock_data_type"
    child1_field_type._fields_ = [(gchild_field_name, gchild_field_type)]

    parent_field_type.__str__ = MagicMock()
    parent_field_type.__str__.return_value = "message_headers.mock_data_type"
    parent_field_type._fields_ = [(child1_field_name, child1_field_type),
                                               (child2_field_name, child2_field_type)]

    # act
    result = cut.gather_field_names(parent_field_name, parent_field_type)

    # assert
    assert isinstance(result, list)
    assert len(result) == 2
    print(result)
    assert set(result) == set([parent_field_name + '.' + child2_field_name, 
                               parent_field_name + '.' + child1_field_name+ '.' +gchild_field_name])

# parse_meta_data_file tests
# TODO
def test_sbn_adapter_DataSource_parse_meta_data_file_calls_rasies_ConfigKeyError_when_channels_not_in_config(mocker):
    # Arrange
    cut = DataSource.__new__(DataSource)
    arg_meta_data_file = MagicMock()
    arg_ss_breakdown = MagicMock()

    mocker.patch(sbn_adapter.__name__ + '.json.loads', return_value = {})

    # Act
    with pytest.raises(ConfigKeyError) as e_info:
        cut.parse_meta_data_file(arg_meta_data_file,arg_ss_breakdown)

def test_sbn_adapter_DataSource_parse_meta_data_file_populates_lookup_table_and_current_data_on_ideal_config(mocker):
    # Arrange
    cut = DataSource.__new__(DataSource)
    arg_meta_data_file = MagicMock()
    arg_ss_breakdown = MagicMock()

    ideal_config = {
        "channels": {
            "0x1": ["AppName1", "DataStruct1"],
            "0x2": ["AppName2", "DataStruct2"]
        }
    }
    
    mock_struct_1 = MagicMock()
    mock_struct_1.__name__ = "DataStruct1"
    mock_struct_1._fields_ = [('TlmHeader', 'type0'), ('field1', 'type1')]

    mock_struct_2 = MagicMock()
    mock_struct_2.__name__ = "DataStruct2"
    mock_struct_2._fields_ = [('field0', 'type0'), ('field1', 'type1')]

    mocker.patch('message_headers.DataStruct1', mock_struct_1)
    mocker.patch('message_headers.DataStruct2', mock_struct_2)

    mocker.patch('builtins.open', mocker.mock_open(read_data=json.dumps(ideal_config)))
    mocker.patch('json.loads', return_value=ideal_config)
    expected_configs = MagicMock()
    mocker.patch(sbn_adapter.__name__ + '.extract_meta_data_handle_ss_breakdown', return_value = expected_configs)

    # Act
    cut.parse_meta_data_file(arg_meta_data_file, arg_ss_breakdown)
    print(cut.currentData)

    # Assert
    assert cut.msgID_lookup_table == {1: ['AppName1', mock_struct_1], 2: ['AppName2', mock_struct_2]}
    assert len(cut.currentData) == 2
    assert len(cut.currentData[0]['headers']) == 2
    assert len(cut.currentData[1]['headers']) == 2
    assert cut.currentData[0]['headers'] == ['AppName1.field1', 'AppName2.field1']
    assert cut.currentData[0]['data'] == [[0], [0]]
    assert cut.currentData[1]['headers'] == ['AppName1.field1', 'AppName2.field1']
    assert cut.currentData[1]['data'] == [[0], [0]]

# process_data_file tests
def test_sbn_adapter_DataSource_process_data_file_does_nothing(mocker):
    # copied from test_redis_adapter.py
    # test_redis_adapter_DataSource_process_data_file_does_nothing
    cut = DataSource.__new__(DataSource)
    arg_data_file = MagicMock()

    expected_result = None

    # Act
    result = cut.process_data_file(arg_data_file)

    # Assert
    assert result == expected_result

# get_vehicle_metadata tests
def test_sbn_adapter_DataSource_get_vehicle_metadata_returns_list_of_headers_and_list_of_test_assignments():
    # copied from test_redis_adapter.py
    # test_redis_adapter_DataSource_get_vehicle_metadata_returns_list_of_headers_and_list_of_test_assignments
    
    # Arrange
    cut = DataSource.__new__(DataSource)
    fake_all_headers = MagicMock()
    fake_test_assignments = MagicMock()
    fake_binning_configs = {}
    fake_binning_configs['test_assignments'] = fake_test_assignments

    expected_result = (fake_all_headers, fake_test_assignments)

    cut.all_headers = fake_all_headers
    cut.binning_configs = fake_binning_configs

    # Act
    result = cut.get_vehicle_metadata()

    # Assert
    assert result == expected_result


# get_next tests
def test_sbn_adapter_DataSource_get_next_returns_expected_data_when_new_data_is_true_and_double_buffer_read_index_is_0():
    # copied from test_redis_adapter.py
    # test_redis_adapter_DataSource_get_next_returns_expected_data_when_new_data_is_true_and_double_buffer_read_index_is_0

    # Arrange
    # Renew DataSource to ensure test independence
    cut = DataSource.__new__(DataSource)
    cut.new_data = True
    cut.new_data_lock = MagicMock()
    cut.double_buffer_read_index = 0
    pre_call_index = cut.double_buffer_read_index
    expected_result = MagicMock()
    cut.currentData = []
    cut.currentData.append({'data': MagicMock()})
    cut.currentData.append({'data': expected_result})

    # Act
    result = cut.get_next()

    # Assert
    assert cut.new_data == False
    assert cut.double_buffer_read_index == 1
    assert result == expected_result

def test_sbn_adapter_DataSource_get_next_returns_expected_data_when_new_data_is_true_and_double_buffer_read_index_is_1():
    # copied from test_redis_adapter.py
    # test_redis_adapter_DataSource_get_next_returns_expected_data_when_new_data_is_true_and_double_buffer_read_index_is_1

    # Arrange
    # Renew DataSource to ensure test independence
    cut = DataSource.__new__(DataSource)
    cut.new_data = True
    cut.new_data_lock = MagicMock()
    cut.double_buffer_read_index = 1
    pre_call_index = cut.double_buffer_read_index
    expected_result = MagicMock()
    cut.currentData = []
    cut.currentData.append({'data': expected_result})
    cut.currentData.append({'data': MagicMock()})

    # Act
    result = cut.get_next()

    # Assert
    assert cut.new_data == False
    assert cut.double_buffer_read_index == 0
    assert result == expected_result

def test_sbn_adapter_DataSource_get_next_when_called_multiple_times_when_new_data_is_true():
    # copied from test_redis_adapter.py
    # test_redis_adapter_DataSource_get_next_when_called_multiple_times_when_new_data_is_true
    
    # Arrange
    # Renew DataSource to ensure test independence
    cut = DataSource.__new__(DataSource)
    cut.double_buffer_read_index = pytest.gen.randint(0,1)
    cut.new_data_lock = MagicMock()
    cut.currentData = [MagicMock(), MagicMock()]
    pre_call_index = cut.double_buffer_read_index
    expected_data = []

    # Act
    results = []
    num_calls = pytest.gen.randint(2,10) # arbitrary, 2 to 10
    for i in range(num_calls):
        cut.new_data = True
        fake_new_data = MagicMock()
        if cut.double_buffer_read_index == 0:
            cut.currentData[1] = {'data': fake_new_data}
        else:
            cut.currentData[0] = {'data': fake_new_data}
        expected_data.append(fake_new_data)
        results.append(cut.get_next())

    # Assert
    assert cut.new_data == False
    for i in range(num_calls):
        results[i] = expected_data[i]
    assert cut.double_buffer_read_index == (num_calls + pre_call_index) % 2

def test_sbn_adapter_DataSource_get_next_waits_until_new_data_is_available(mocker):
    # copied from test_redis_adapter.py
    # test_redis_adapter_DataSource_get_next_waits_until_new_data_is_available
    
    # Arrange
    # Renew DataSource to ensure test independence
    cut = DataSource.__new__(DataSource)
    cut.new_data_lock = MagicMock()
    cut.double_buffer_read_index = pytest.gen.randint(0,1)
    pre_call_index = cut.double_buffer_read_index
    expected_result = MagicMock()
    cut.new_data = None
    cut.currentData = []
    if pre_call_index == 0:
        cut.currentData.append({'data': MagicMock()})
        cut.currentData.append({'data': expected_result})
    else:
        cut.currentData.append({'data': expected_result})
        cut.currentData.append({'data': MagicMock()})

    num_falses = pytest.gen.randint(1, 10)
    side_effect_list = [False] * num_falses
    side_effect_list.append(True)

    mocker.patch.object(cut, 'has_data', side_effect=side_effect_list)
    mocker.patch(sbn_adapter.__name__ + '.time.sleep')

    # Act
    result = cut.get_next()

    # Assert
    assert cut.has_data.call_count == num_falses + 1
    assert sbn_adapter.time.sleep.call_count == num_falses
    assert cut.new_data == False
    if pre_call_index == 0:
        assert cut.double_buffer_read_index == 1
    elif pre_call_index == 1:
        assert cut.double_buffer_read_index == 0
    else:
        assert False

    assert result == expected_result

# has_more tests
def test_sbn_adapter_DataSource_has_more_always_returns_True():
    # copied from test_redis_adapter.py
    # test_redis_adapter_DataSource_has_more_always_returns_True
    cut = DataSource.__new__(DataSource)
    assert cut.has_more() == True

# mesage_listener_thread tests
# TODO

# get_current_data tests
def test_sbn_adapter_Data_Source_get_current_data_only_changes_time_data_when_no_fields_present_in_msg(mocker):
    # Arrange
    cut = DataSource.__new__(DataSource)
    cut.double_buffer_read_index = pytest.gen.randint(0,1)
    n = pytest.gen.randint(1,9)
    cut.currentData =  [{'headers':[f'field_{i}' for i in range(n)],'data':[[0] for x in range(n)]}, 
                        {'headers':[f'field_{i}' for i in range(n)],'data':[[0] for x in range(n)]}]
    cut.new_data_lock = MagicMock()

    arg_recv_msg = MagicMock()
    arg_recv_msg._fields_ = ['header'] # other fields would populate indicies 1 and up. No actual fields in the struct 
    arg_recv_msg.TlmHeader.Secondary = MagicMock()
    arg_recv_msg.TlmHeader.Secondary.Seconds = pytest.gen.randint(0,9)
    arg_recv_msg.TlmHeader.Secondary.Subseconds = pytest.gen.randint(0,9)

    arg_data_struct = MagicMock()
    arg_app_name = MagicMock()

    start_time = datetime.datetime(1969, 12, 31, 20)
    seconds = arg_recv_msg.TlmHeader.Secondary.Seconds
    subseconds = arg_recv_msg.TlmHeader.Secondary.Subseconds
    curr_time = seconds + (2**(-32) * subseconds)
    time = start_time + datetime.timedelta(seconds=curr_time)
    str_time = time.strftime("%Y-%j-%H:%M:%S.%f")

    expected_currentData = copy.deepcopy(cut.currentData)
    expected_currentData[(cut.double_buffer_read_index + 1) % 2]['data'][0] = str_time

    # Act
    result = cut.get_current_data(arg_recv_msg, arg_data_struct, arg_app_name)

    # Assert
    assert result is None
    assert isinstance(cut.currentData, list)
    assert cut.currentData == expected_currentData
    assert isinstance(cut.new_data, bool)
    assert cut.new_data 

def test_sbn_adapter_Data_Source_get_current_data_calls_gather_field_names_correctly(mocker):
    # Arrange
    cut = DataSource.__new__(DataSource)
    cut.double_buffer_read_index = pytest.gen.randint(0,1)
    n = pytest.gen.randint(1,9)
    cut.currentData =  [{'headers':[f'field_{i}' for i in range(n)],'data':[[0] for x in range(n)]}, 
                        {'headers':[f'field_{i}' for i in range(n)],'data':[[0] for x in range(n)]}]
    cut.new_data_lock = MagicMock()

    arg_recv_msg = MagicMock()
    arg_recv_msg._fields_ = [(MagicMock(), MagicMock()) for x in range(n)]
    arg_recv_msg._fields_.insert(0, 'header')
    arg_recv_msg.TlmHeader.Secondary = MagicMock()
    arg_recv_msg.TlmHeader.Secondary.Seconds = pytest.gen.randint(0,9)
    arg_recv_msg.TlmHeader.Secondary.Subseconds = pytest.gen.randint(0,9)

    arg_data_struct = MagicMock()
    arg_app_name = MagicMock()
    
    mocker.patch.object(cut, 'gather_field_names', return_value = [])

    # act
    cut.get_current_data(arg_recv_msg, arg_data_struct, arg_app_name)

    # Assert
    assert cut.gather_field_names.call_count == n
    assert len(cut.gather_field_names.call_args_list) == n
    for i in range(n):
        expected_args = arg_recv_msg._fields_[i+1]
        assert cut.gather_field_names.call_args_list[i].args == expected_args

def test_sbn_adapter_DataSource_get_current_data_unpacks_sub_fields_correctly(mocker):
    # TODO
    # Arrange
    cut = DataSource.__new__(DataSource)
    cut.double_buffer_read_index = pytest.gen.randint(0,1)
    n = pytest.gen.randint(1,9)
    cut.currentData =  [{'headers':[f'field_{i}' for i in range(n)],'data':[[0] for x in range(n)]}, 
                        {'headers':[f'field_{i}' for i in range(n)],'data':[[0] for x in range(n)]}]
    cut.new_data_lock = MagicMock()

    arg_recv_msg = MagicMock()
    arg_recv_msg._fields_ = [(MagicMock(), MagicMock()) for x in range(n)]
    arg_recv_msg._fields_.insert(0, 'header')
    arg_recv_msg.TlmHeader.Secondary = MagicMock()
    arg_recv_msg.TlmHeader.Secondary.Seconds = pytest.gen.randint(0,9)
    arg_recv_msg.TlmHeader.Secondary.Subseconds = pytest.gen.randint(0,9)

    arg_data_struct = MagicMock()
    arg_app_name = 'mock_app'
    
    mocker.patch.object(cut, 'gather_field_names', return_value = [])


# has_data tests
def test_sbn_adapter_DataSource_has_data_returns_instance_new_data():
    # copied from test_redis_adapter.py
    # test_redis_adapter_DataSource_has_data_returns_instance_new_data
    cut = DataSource.__new__(DataSource)
    expected_result = MagicMock()
    cut.new_data = expected_result

    result = cut.has_data()

    assert result == expected_result
