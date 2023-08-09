# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

import pytest
from mock import MagicMock, PropertyMock
import onair.src.run_scripts.sbn_adapter as sbn_adapter
from importlib import reload
import sys

@pytest.fixture
def setup_teardown():
    # # refresh sbn_adapter module before each test to ensure independence 
    reload(sbn_adapter)
    
    print('setup')
    
    yield 'setup_teardown'

    print('teardown')

    # refresh message_headers module after each test to remove any changes from testing
    del sys.modules['message_headers']
    mh = MagicMock()
    mh.sample_data_tlm_t = MagicMock()
    mh.sample_data_tlm_t.__name__ = 'mock_sample_data_tlm_t'
    mh.sample_data_power_t = MagicMock()
    mh.sample_data_power_t.__name__ = 'mock_sample_data_power_t'
    mh.sample_data_thermal_t = MagicMock()
    mh.sample_data_thermal_t.__name__ = 'mock_sample_data_thermal_t'
    mh.sample_data_gps_t = MagicMock()
    mh.sample_data_gps_t.__name__ = 'mock_sample_data_gps_t'
    sys.modules['message_headers'] = mh


class FakeSbnDataGenericT(MagicMock):
    pass

class FakeDataStruct(MagicMock):
    pass

# testing that the msgID_lookup_table is as we expect allows us to safely reference it in other tests
def test_sbn_adapter_msgID_lookup_table_is_expected_value():
    # Arrange
    expected_msgID_lookup_table = {0x0885 : ["SAMPLE", sbn_adapter.msg_hdr.sample_data_tlm_t],
                                   0x0887 : ["SAMPLE", sbn_adapter.msg_hdr.sample_data_power_t],
                                   0x0889 : ["SAMPLE", sbn_adapter.msg_hdr.sample_data_thermal_t],
                                   0x088A : ["SAMPLE", sbn_adapter.msg_hdr.sample_data_gps_t]}

    # Act
    # Assert
    assert sbn_adapter.msgID_lookup_table == expected_msgID_lookup_table

def test_sbn_adapter_message_listener_thread_loops_indefinitely_until_purposely_broken(mocker, setup_teardown):
    # Arrange
    fake_generic_recv_msg_p = MagicMock()
    fake_recv_msg_p = MagicMock()

    fake_sbn = MagicMock()
    fake_sbn.sbn_data_generic_t = PropertyMock()

    fake_generic_recv_msg_p_contents = PropertyMock()
    fake_generic_recv_msg_p_TlmHeader = PropertyMock()
    fake_generic_recv_msg_p_Primary = PropertyMock()
    
    fake_generic_recv_msg_p.contents = fake_generic_recv_msg_p_contents
    fake_generic_recv_msg_p.contents.TlmHeader = fake_generic_recv_msg_p_TlmHeader
    fake_generic_recv_msg_p.contents.TlmHeader.Primary = fake_generic_recv_msg_p_Primary
    fake_msgID = pytest.gen.choice(list(sbn_adapter.msgID_lookup_table.keys()))
    fake_generic_recv_msg_p.contents.TlmHeader.Primary.StreamId = fake_msgID

    # this exception will be used to forcefully exit the message_listener_thread function's while(True) loop
    exception_message = 'forced loop exit'
    exit_exception = Exception(exception_message)
    
    # sets return value of POINTER function to return fake_pointer an arbitrary number of times, then return exit_exception
    num_loop_iterations = pytest.gen.randint(1, 10) # arbitrary, 1 to 10
    side_effect_list = ([''] * num_loop_iterations) # one item for each loop
    side_effect_list.append(exit_exception) # short-circuit exit while(True) loop

    fake__fields_ = [["1st item placeholder"]]
    num_fake_prints = pytest.gen.randint(1, 10) # arbitrary from 1 to 10
    fake_field_names = []
    fake_attr_values = []
    expected_print_string = ''
    for i in range(num_fake_prints):
        fake_attr_name = str(MagicMock())

        print(f"fake_attr_name ", i, " ", fake_attr_name)
        fake_attr_value = MagicMock()

        print(f"fake_attr_value ", i, " ", fake_attr_value)
        fake_field_names.append(fake_attr_name)
        fake_attr_values.append(fake_attr_value)
        fake__fields_.append([fake_attr_name, fake_attr_value])
        expected_print_string  += fake_attr_name + ": " + str(fake_attr_value) + ", " 

    fake_generic_recv_msg_p_contents._fields_ = fake__fields_
    expected_print_string = expected_print_string[0:-2]

    mocker.patch('onair.src.run_scripts.sbn_adapter.sbn', fake_sbn)
    pointer_side_effects = [FakeSbnDataGenericT, FakeDataStruct] * (num_loop_iterations + 1)
    mocker.patch('onair.src.run_scripts.sbn_adapter.POINTER', side_effect=pointer_side_effects)
    mocker.patch.object(FakeSbnDataGenericT, '__new__', return_value=fake_generic_recv_msg_p)
    mocker.patch.object(fake_sbn, 'recv_msg', return_value=None)
    mocker.patch.object(FakeDataStruct, '__new__', return_value=fake_recv_msg_p)
    mocker.patch('onair.src.run_scripts.sbn_adapter.getattr', side_effect=fake_attr_values * (num_loop_iterations + 1))
    mocker.patch('onair.src.run_scripts.sbn_adapter.print', return_value=None)

    mocker.patch('onair.src.run_scripts.sbn_adapter.get_current_data', side_effect=side_effect_list)
    
    # Act
    with pytest.raises(Exception) as e_info:
        sbn_adapter.message_listener_thread()

    # Assert
    assert e_info.match(exception_message)
    assert sbn_adapter.POINTER.call_count == (num_loop_iterations + 1) * 2
    for i in range(num_loop_iterations + 1):
        assert sbn_adapter.POINTER.call_args_list[i*2].args == (fake_sbn.sbn_data_generic_t, )
        assert sbn_adapter.POINTER.call_args_list[(i*2)+1].args == (sbn_adapter.msgID_lookup_table[fake_msgID][1], )
    assert FakeSbnDataGenericT.__new__.call_count == num_loop_iterations + 1
    assert FakeDataStruct.__new__.call_count == num_loop_iterations + 1
    assert fake_sbn.recv_msg.call_count == num_loop_iterations + 1
    for i in range(num_loop_iterations + 1):
        assert fake_sbn.recv_msg.call_args_list[i].args == (fake_generic_recv_msg_p, )
    assert sbn_adapter.getattr.call_count == (num_loop_iterations + 1) * num_fake_prints
    for i in range((num_loop_iterations + 1) * num_fake_prints):
        assert sbn_adapter.getattr.call_args_list[i].args == (fake_generic_recv_msg_p_contents, fake_field_names[i % len(fake_field_names)])
    assert sbn_adapter.print.call_count == num_loop_iterations + 1
    for i in range(num_loop_iterations + 1):
        assert sbn_adapter.print.call_args_list[i].args == (expected_print_string, )
    assert sbn_adapter.get_current_data.call_count == num_loop_iterations + 1
    for i in range(num_loop_iterations + 1):
        assert sbn_adapter.get_current_data.call_args_list[i].args == (fake_generic_recv_msg_p_contents, sbn_adapter.msgID_lookup_table[fake_msgID][1], sbn_adapter.msgID_lookup_table[fake_msgID][0])
  
def test_get_current_data_with_no_fields_in_recv_msg_or_data_struct(mocker, setup_teardown):
    # Arrange
    fake_generic_recv_msg_p = MagicMock()

    fake_AdapterDataSource = MagicMock
    fake_AdapterDataSource.currentData = {}
    fake_AdapterDataSource.double_buffer_read_index = 1
    fake_AdapterDataSource.new_data_lock = PropertyMock()
    fake_current_buffer = {}
    fake_current_buffer['data'] = ['placeholder']

    fake_AdapterDataSource.currentData[0] = fake_current_buffer

    arg_recv_msg = PropertyMock()
    fake_generic_recv_msg_p_TlmHeader = PropertyMock()
    fake_generic_recv_msg_p_Primary = PropertyMock()
    fake_recv_msg_p_Secondary = PropertyMock()
    
    fake_generic_recv_msg_p.contents = arg_recv_msg
    fake_generic_recv_msg_p.contents.TlmHeader = fake_generic_recv_msg_p_TlmHeader
    fake_generic_recv_msg_p.contents.TlmHeader.Primary = fake_generic_recv_msg_p_Primary
    fake_msgID = pytest.gen.choice(list(sbn_adapter.msgID_lookup_table.keys()))
    fake_generic_recv_msg_p.contents.TlmHeader.Primary.StreamId = fake_msgID
    arg_recv_msg.TlmHeader.Secondary = fake_recv_msg_p_Secondary
    fake_seconds = pytest.gen.randint(0,59)
    fake_recv_msg_p_Secondary.Seconds = fake_seconds
    fake_subseconds = pytest.gen.randint(0,999)
    fake_recv_msg_p_Secondary.Subseconds = fake_subseconds
    fake_start_time = MagicMock()
    fake_timedelta = MagicMock()
    fake_time = MagicMock()
    fake_str_time = MagicMock()

    mocker.patch('onair.src.run_scripts.sbn_adapter.AdapterDataSource',fake_AdapterDataSource)
    mocker.patch('onair.src.run_scripts.sbn_adapter.datetime.datetime', return_value=fake_start_time)
    mocker.patch('onair.src.run_scripts.sbn_adapter.datetime.timedelta', return_value=fake_timedelta)
    mocker.patch.object(fake_start_time, '__add__', return_value=fake_time)
    mocker.patch.object(fake_time, 'strftime', return_value=fake_str_time)
    fake_AdapterDataSource.new_data = False # not required, but helps verify it gets changed to True
    mocker.patch.object(fake_AdapterDataSource.new_data_lock, '__enter__') # __enter__ is used by keyword 'with' in python

    arg_data_struct = sbn_adapter.msgID_lookup_table[fake_msgID][1]
    arg_app_name = sbn_adapter.msgID_lookup_table[fake_msgID][0]

    # Act
    sbn_adapter.get_current_data(arg_recv_msg, arg_data_struct, arg_app_name)

    # Assert
    assert sbn_adapter.datetime.datetime.call_count ==  1
    assert sbn_adapter.datetime.datetime.call_args_list[0].args == (1969, 12, 31, 20)
    assert sbn_adapter.datetime.timedelta.call_count == 1
    assert sbn_adapter.datetime.timedelta.call_args_list[0].kwargs == {'seconds':fake_seconds  + (2**(-32) * fake_subseconds)}
    # Although patched and does return value, fake_start_time.__add__ does not count but leaving comments here to show we would do this if able
    # assert fake_start_time.__add__.call_count == num_loop_iterations + 1
    # for i in range(num_loop_iterations + 1):
    #     assert fake_start_time.__add__.call_args_list[i].args == (fake_timedelta, )
    assert fake_time.strftime.call_count == 1
    assert fake_time.strftime.call_args_list[0].args == ("%Y-%j-%H:%M:%S.%f", )
    assert fake_AdapterDataSource.new_data == True
    assert fake_current_buffer['data'][0] == fake_str_time
    
def test_get_current_data_with_fields_in_recv_msg_and_data_struct(mocker, setup_teardown):
    # Arrange
    fake_generic_recv_msg_p = MagicMock()

    fake_AdapterDataSource = MagicMock()
    fake_AdapterDataSource.currentData = {}
    fake_AdapterDataSource.double_buffer_read_index = 1
    fake_AdapterDataSource.new_data_lock = PropertyMock()
    fake_AdapterDataSource.new_data_lock.__enter__ = MagicMock()
    fake_AdapterDataSource.new_data_lock.__exit__ = MagicMock()
    fake_current_buffer = {}
    fake_current_buffer['data'] = ['placeholder0', 'placeholder1']
    fake_headers_for_current_buffer = MagicMock()
    fake_current_buffer['headers'] = fake_headers_for_current_buffer
    fake_idx = 1
    fake_AdapterDataSource.currentData[0] = fake_current_buffer
    arg_recv_msg = PropertyMock()
    fake_generic_recv_msg_p_TlmHeader = PropertyMock()
    fake_generic_recv_msg_p_Primary = PropertyMock()
    fake_recv_msg_p_Secondary = PropertyMock()
    
    fake_generic_recv_msg_p.contents = arg_recv_msg
    fake_generic_recv_msg_p.contents.TlmHeader = fake_generic_recv_msg_p_TlmHeader
    fake_generic_recv_msg_p.contents.TlmHeader.Primary = fake_generic_recv_msg_p_Primary
    fake_msgID = pytest.gen.choice(list(sbn_adapter.msgID_lookup_table.keys()))
    fake_generic_recv_msg_p.contents.TlmHeader.Primary.StreamId = fake_msgID
    arg_recv_msg.TlmHeader.Secondary = fake_recv_msg_p_Secondary
    fake_seconds = pytest.gen.randint(0,59)
    fake_recv_msg_p_Secondary.Seconds = fake_seconds
    fake_subseconds = pytest.gen.randint(0,999)
    fake_recv_msg_p_Secondary.Subseconds = fake_subseconds
    fake_start_time = MagicMock()
    fake_timedelta = MagicMock()
    fake_time = MagicMock()
    fake_str_time = MagicMock()

    fake__fields_ = [["1st item placeholder"]]
    num_fake__fields_ = pytest.gen.randint(1, 10) # arbitrary from 1 to 10
    fake_field_names = []
    for i in range(num_fake__fields_):
        fake_attr_name = str(MagicMock())
        fake_attr_value = MagicMock()
        
        fake_field_names.append(fake_attr_name)
        arg_recv_msg.__setattr__(fake_attr_name, fake_attr_value)
        sbn_adapter.msgID_lookup_table[fake_msgID][1].__setattr__(fake_attr_name, fake_attr_value)
        fake__fields_.append([fake_attr_name, fake_attr_value])

    arg_recv_msg._fields_ = fake__fields_
    sbn_adapter.msgID_lookup_table[fake_msgID][1]._fields_ = fake__fields_

    mocker.patch('onair.src.run_scripts.sbn_adapter.AdapterDataSource',fake_AdapterDataSource)
    mocker.patch('onair.src.run_scripts.sbn_adapter.datetime.datetime', return_value=fake_start_time)
    mocker.patch('onair.src.run_scripts.sbn_adapter.datetime.timedelta', return_value=fake_timedelta)
    mocker.patch.object(fake_start_time, '__add__', return_value=fake_time)
    mocker.patch.object(fake_time, 'strftime', return_value=fake_str_time)
    fake_AdapterDataSource.new_data = False # not required, but helps verify it gets changed to True
    mocker.patch.object(fake_AdapterDataSource.new_data_lock, '__enter__') # __enter__ is used by keyword 'with' in python
    mocker.patch.object(fake_headers_for_current_buffer, 'index', return_value=fake_idx)

    arg_data_struct = sbn_adapter.msgID_lookup_table[fake_msgID][1]
    arg_app_name = sbn_adapter.msgID_lookup_table[fake_msgID][0]

    # Act
    sbn_adapter.get_current_data(arg_recv_msg, arg_data_struct, arg_app_name)

    # Assert
    assert sbn_adapter.datetime.datetime.call_count == 1
    assert sbn_adapter.datetime.datetime.call_args_list[0].args == (1969, 12, 31, 20)
    assert sbn_adapter.datetime.timedelta.call_count == 1
    assert sbn_adapter.datetime.timedelta.call_args_list[0].kwargs == {'seconds':fake_seconds  + (2**(-32) * fake_subseconds)}
    # Although patched and does return value, fake_start_time.__add__ does not count but leaving comments here to show we would do this if able
    # assert fake_start_time.__add__.call_count == num_loop_iterations + 1
    # for i in range(num_loop_iterations + 1):
    #     assert fake_start_time.__add__.call_args_list[i].args == (fake_timedelta, )
    assert fake_time.strftime.call_count == 1
    assert fake_time.strftime.call_args_list[0].args == ("%Y-%j-%H:%M:%S.%f", )
    assert fake_headers_for_current_buffer.index.call_count == num_fake__fields_
    for i in range(num_fake__fields_):
        assert fake_headers_for_current_buffer.index.call_args_list[i].args == (str((sbn_adapter.msgID_lookup_table[fake_msgID][0]) + "." + str(sbn_adapter.msgID_lookup_table[fake_msgID][1].__name__) + "." + fake_field_names[i]), )
    assert fake_AdapterDataSource.new_data == True
    assert fake_current_buffer['data'][0] == fake_str_time

# ---------- Tests for AdapterDataSource class ---------

# tests for AdapterDataSource variables
def test_sbn_adapter_AdapterDataSource_current_data_equals_expected_value_when_no__fields__exist_in_data_struct(setup_teardown):
    # Arrange
    # Renew AdapterDataSource to ensure test independence
    AdapterDataSource = sbn_adapter.AdapterDataSource
    lookup_table = sbn_adapter.msgID_lookup_table

    expected_current_data = []

    for x in range(2):
        expected_current_data.append({'headers' : [], 'data' : []})
        expected_current_data[x]['headers'].append('TIME')
        expected_current_data[x]['data'].append('2000-001-12:00:00.000000000')

        for msgID in lookup_table.keys():
            app_name, data_struct = lookup_table[msgID]
            struct_name = data_struct.__name__
            expected_current_data[x]['data'].extend([0]*len(data_struct._fields_[1:])) #initialize all the data arrays with zero

    # Act
    cut = AdapterDataSource.__new__(AdapterDataSource)

    # Assert
    assert cut.currentData == expected_current_data

def test_sbn_adapter_AdapterDataSource_current_data_equals_expected_value_when__fields___do_exist_in_data_struct(setup_teardown):
    # Arrange
    lookup_table = sbn_adapter.msgID_lookup_table

    expected_current_data = []

    for x in range(2):
        expected_current_data.append({'headers' : [], 'data' : []})
        expected_current_data[x]['headers'].append('TIME')
        expected_current_data[x]['data'].append('2000-001-12:00:00.000000000')
        for msgID in lookup_table.keys():
            app_name, data_struct = lookup_table[msgID]
            struct_name = data_struct.__name__

            fake__fields_ = [["1st item placeholder"]]
            num_fake__fields_ = 1#pytest.gen.randint(1, 10) # arbitrary from 1 to 10
            fake_field_names = []
            
            for i in range(num_fake__fields_):
                fake_attr_name = "fake_attr_name"
                fake_attr_value = MagicMock()
                fake_field_names.append(fake_attr_name)
                fake__fields_.append([fake_attr_name, fake_attr_value])

            data_struct._fields_ = fake__fields_
            sbn_adapter.msgID_lookup_table[msgID][1].__setattr__('_fields_', fake__fields_)
                
            for field_name, field_type in data_struct._fields_[1:]:
                expected_current_data[x]['headers'].append('{}.{}.{}'.format(app_name, struct_name, str(field_name)))
    
            expected_current_data[x]['data'].extend([0]*num_fake__fields_) #initialize all the data arrays with zero
    
    # Renew abn_adapter and AdapterDataSource to ensure test independence and get field updates to message_headers module
    reload(sbn_adapter)
    AdapterDataSource = sbn_adapter.AdapterDataSource
    
    # Act
    cut = AdapterDataSource.__new__(AdapterDataSource)

    # Assert
    assert cut.currentData == expected_current_data

# tests for AdapterDataSource.connect
def test_sbn_adapter_AdapterDataSource_connect_when_msgID_lookup_table_has_zero_keys(mocker, setup_teardown):
    # Arrange
    # Renew AdapterDataSource to ensure test independence
    AdapterDataSource = sbn_adapter.AdapterDataSource
    fake_listener_thread = MagicMock()
    fake_msgID_lookup_table = MagicMock()
    fake_msgID_lookup_table_keys = []

    mocker.patch('onair.src.run_scripts.sbn_adapter.time.sleep')
    mocker.patch('onair.src.run_scripts.sbn_adapter.os.chdir')
    mocker.patch('onair.src.run_scripts.sbn_adapter.sbn.sbn_load_and_init')
    mocker.patch('onair.src.run_scripts.sbn_adapter.message_listener_thread', fake_listener_thread)
    mocker.patch('onair.src.run_scripts.sbn_adapter.threading.Thread', return_value=fake_listener_thread)
    mocker.patch.object(fake_listener_thread, 'start')
    mocker.patch('onair.src.run_scripts.sbn_adapter.msgID_lookup_table', fake_msgID_lookup_table)
    mocker.patch.object(fake_msgID_lookup_table, 'keys', return_value=fake_msgID_lookup_table_keys)
    mocker.patch('onair.src.run_scripts.sbn_adapter.sbn.subscribe')
    
    cut = AdapterDataSource.__new__(AdapterDataSource)

    # Act
    cut.connect()

    # Assert
    assert sbn_adapter.time.sleep.call_count == 1
    assert sbn_adapter.time.sleep.call_args_list[0].args == (2,)
    assert sbn_adapter.os.chdir.call_count == 1
    assert sbn_adapter.os.chdir.call_args_list[0].args == ('cf',)
    assert sbn_adapter.sbn.sbn_load_and_init.call_count == 1
    
    assert sbn_adapter.threading.Thread.call_count == 1
    assert sbn_adapter.threading.Thread.call_args_list[0].args == ()
    assert sbn_adapter.threading.Thread.call_args_list[0].kwargs == {'target' : fake_listener_thread}
    assert fake_listener_thread.start.call_count == 1

    assert fake_msgID_lookup_table.keys.call_count == 1
    assert sbn_adapter.sbn.subscribe.call_count == 0

def test_sbn_adapter_AdapterDataSource_connect_when_msgID_lookup_table_has_one_key(mocker, setup_teardown):
    # Arrange
    # Renew AdapterDataSource to ensure test independence
    AdapterDataSource = sbn_adapter.AdapterDataSource
    fake_listener_thread = MagicMock()
    fake_msgID_lookup_table = MagicMock()
    fake_msgID = MagicMock()
    fake_msgID_lookup_table_keys = [fake_msgID]

    mocker.patch('onair.src.run_scripts.sbn_adapter.time.sleep')
    mocker.patch('onair.src.run_scripts.sbn_adapter.os.chdir')
    mocker.patch('onair.src.run_scripts.sbn_adapter.sbn.sbn_load_and_init')
    mocker.patch('onair.src.run_scripts.sbn_adapter.message_listener_thread', fake_listener_thread)
    mocker.patch('onair.src.run_scripts.sbn_adapter.threading.Thread', return_value=fake_listener_thread)
    mocker.patch.object(fake_listener_thread, 'start')
    mocker.patch('onair.src.run_scripts.sbn_adapter.msgID_lookup_table', fake_msgID_lookup_table)
    mocker.patch.object(fake_msgID_lookup_table, 'keys', return_value=fake_msgID_lookup_table_keys)
    mocker.patch('onair.src.run_scripts.sbn_adapter.sbn.subscribe')
    
    cut = AdapterDataSource.__new__(AdapterDataSource)

    # Act
    cut.connect()

    # Assert
    assert sbn_adapter.time.sleep.call_count == 1
    assert sbn_adapter.time.sleep.call_args_list[0].args == (2,)
    assert sbn_adapter.os.chdir.call_count == 1
    assert sbn_adapter.os.chdir.call_args_list[0].args == ('cf',)
    assert sbn_adapter.sbn.sbn_load_and_init.call_count == 1
    
    assert sbn_adapter.threading.Thread.call_count == 1
    assert sbn_adapter.threading.Thread.call_args_list[0].args == ()
    assert sbn_adapter.threading.Thread.call_args_list[0].kwargs == {'target' : fake_listener_thread}
    assert fake_listener_thread.start.call_count == 1

    assert fake_msgID_lookup_table.keys.call_count == 1
    assert sbn_adapter.sbn.subscribe.call_count == 1
    assert sbn_adapter.sbn.subscribe.call_args_list[0].args == (fake_msgID,)
    
def test_sbn_adapter_AdapterDataSource_connect_when_msgID_lookup_table_has_multiple_keys(mocker, setup_teardown):
    # Arrange
    # Renew AdapterDataSource to ensure test independence
    AdapterDataSource = sbn_adapter.AdapterDataSource
    fake_listener_thread = MagicMock()
    fake_msgID_lookup_table = MagicMock()
    num_keys = pytest.gen.randint(2, 10) # arbitrary, from 2 to 10
    fake_msgID_lookup_table_keys = []
    for i in range(num_keys):
        fake_msgID_lookup_table_keys.append(MagicMock())

    mocker.patch('onair.src.run_scripts.sbn_adapter.time.sleep')
    mocker.patch('onair.src.run_scripts.sbn_adapter.os.chdir')
    mocker.patch('onair.src.run_scripts.sbn_adapter.sbn.sbn_load_and_init')
    mocker.patch('onair.src.run_scripts.sbn_adapter.message_listener_thread', fake_listener_thread)
    mocker.patch('onair.src.run_scripts.sbn_adapter.threading.Thread', return_value=fake_listener_thread)
    mocker.patch.object(fake_listener_thread, 'start')
    mocker.patch('onair.src.run_scripts.sbn_adapter.msgID_lookup_table', fake_msgID_lookup_table)
    mocker.patch.object(fake_msgID_lookup_table, 'keys', return_value=fake_msgID_lookup_table_keys)
    mocker.patch('onair.src.run_scripts.sbn_adapter.sbn.subscribe')
    
    cut = AdapterDataSource.__new__(AdapterDataSource)

    # Act
    cut.connect()

    # Assert
    assert sbn_adapter.time.sleep.call_count == 1
    assert sbn_adapter.time.sleep.call_args_list[0].args == (2,)
    assert sbn_adapter.os.chdir.call_count == 1
    assert sbn_adapter.os.chdir.call_args_list[0].args == ('cf',)
    assert sbn_adapter.sbn.sbn_load_and_init.call_count == 1
    
    assert sbn_adapter.threading.Thread.call_count == 1
    assert sbn_adapter.threading.Thread.call_args_list[0].args == ()
    assert sbn_adapter.threading.Thread.call_args_list[0].kwargs == {'target' : fake_listener_thread}
    assert fake_listener_thread.start.call_count == 1

    assert fake_msgID_lookup_table.keys.call_count == 1
    assert sbn_adapter.sbn.subscribe.call_count == num_keys
    for i in range(num_keys):
        assert sbn_adapter.sbn.subscribe.call_args_list[i].args == (fake_msgID_lookup_table_keys[i],)
    
# tests for AdapterDataSource.subscribe_message
def test_sbn_adapter_AdapterDataSource_subscribe_message_when_msgid_is_not_a_list(mocker, setup_teardown):
    # Arrange
    # Renew AdapterDataSource to ensure test independence
    AdapterDataSource = sbn_adapter.AdapterDataSource
    arg_msgid = MagicMock()
    
    mocker.patch('onair.src.run_scripts.sbn_adapter.sbn.subscribe')

    cut = AdapterDataSource.__new__(AdapterDataSource)

    # Act
    cut.subscribe_message(arg_msgid)

    # Assert
    assert sbn_adapter.sbn.subscribe.call_count == 1
    assert sbn_adapter.sbn.subscribe.call_args_list[0].args == (arg_msgid,)

def test_sbn_adapter_AdapterDataSource_subscribe_message_when_msgid_is_an_empty_list(mocker, setup_teardown):
    # Arrange
    # Renew AdapterDataSource to ensure test independence
    AdapterDataSource = sbn_adapter.AdapterDataSource
    arg_msgid = []
    
    mocker.patch('onair.src.run_scripts.sbn_adapter.sbn.subscribe')

    cut = AdapterDataSource.__new__(AdapterDataSource)

    # Act
    cut.subscribe_message(arg_msgid)

    # Assert
    assert sbn_adapter.sbn.subscribe.call_count == 0

def test_sbn_adapter_AdapterDataSource_subscribe_message_when_msgid_is_a_list_with_only_one_element(mocker, setup_teardown):
    # Arrange
    # Renew AdapterDataSource to ensure test independence
    AdapterDataSource = sbn_adapter.AdapterDataSource
    arg_msgid = [MagicMock()]
    
    mocker.patch('onair.src.run_scripts.sbn_adapter.sbn.subscribe')

    cut = AdapterDataSource.__new__(AdapterDataSource)

    # Act
    cut.subscribe_message(arg_msgid)

    # Assert
    assert sbn_adapter.sbn.subscribe.call_count == 1
    assert sbn_adapter.sbn.subscribe.call_args_list[0].args == (arg_msgid[0],)

def test_sbn_adapter_AdapterDataSource_subscribe_message_when_msgid_is_a_list_of_multiple_elements(mocker, setup_teardown):
    # Arrange
    # Renew AdapterDataSource to ensure test independence
    AdapterDataSource = sbn_adapter.AdapterDataSource
    list_length = pytest.gen.randint(2,10) # arbitrary, from 2 to 10
    arg_msgid = []
    for i in range(list_length):
        arg_msgid.append(MagicMock())
    
    mocker.patch('onair.src.run_scripts.sbn_adapter.sbn.subscribe')

    cut = AdapterDataSource.__new__(AdapterDataSource)

    # Act
    cut.subscribe_message(arg_msgid)

    # Assert
    assert sbn_adapter.sbn.subscribe.call_count == list_length
    for i in range(list_length):
        assert sbn_adapter.sbn.subscribe.call_args_list[i].args == (arg_msgid[i],)
    
# tests for AdapterDataSource.get_next
def test_sbn_adapter_AdapterDataSource_get_next_when_new_data_is_true(setup_teardown):
    # Arrange
    # Renew AdapterDataSource to ensure test independence
    AdapterDataSource = sbn_adapter.AdapterDataSource
    cut = AdapterDataSource.__new__(AdapterDataSource)
    AdapterDataSource.new_data = True
    pre_call_index = AdapterDataSource.double_buffer_read_index

    # Act
    result = cut.get_next()

    # Assert
    assert AdapterDataSource.new_data == False
    if pre_call_index == 0:
        assert AdapterDataSource.double_buffer_read_index == 1
        assert result == AdapterDataSource.currentData[1]['data']
    elif pre_call_index == 1:
        assert AdapterDataSource.double_buffer_read_index == 0
        assert result == AdapterDataSource.currentData[0]['data']
    else:
        assert False

def test_sbn_adapter_AdapterDataSource_get_next_when_called_multiple_times_when_new_data_is_true(setup_teardown):
    # Arrange
    # Renew AdapterDataSource to ensure test independence
    AdapterDataSource = sbn_adapter.AdapterDataSource
    cut = AdapterDataSource.__new__(AdapterDataSource)
    pre_call_index = AdapterDataSource.double_buffer_read_index

    # Act
    results = []
    num_calls = pytest.gen.randint(2,10) # arbitrary, 2 to 10
    for i in range(num_calls):
        AdapterDataSource.new_data = True
        results.append(cut.get_next())

    # Assert
    assert AdapterDataSource.new_data == False
    for i in range(num_calls):
        results[i] = AdapterDataSource.currentData[pre_call_index]['data']
        pre_call_index = (pre_call_index + 1) % 2
    assert AdapterDataSource.double_buffer_read_index == pre_call_index
    
def test_sbn_adapter_AdapterDataSource_get_next_behavior_when_new_data_is_false_then_true(mocker, setup_teardown):
    # Arrange
    # Renew AdapterDataSource to ensure test independence
    AdapterDataSource = sbn_adapter.AdapterDataSource
    cut = AdapterDataSource.__new__(AdapterDataSource)
    pre_call_index = AdapterDataSource.double_buffer_read_index

    num_falses = pytest.gen.randint(1, 10)
    side_effect_list = [False] * num_falses
    side_effect_list.append(True)

    AdapterDataSource.new_data = PropertyMock(side_effect=side_effect_list)
    mocker.patch('onair.src.run_scripts.sbn_adapter.time.sleep')

    # Act
    result = cut.get_next()

    # Assert
    assert sbn_adapter.time.sleep.call_count == num_falses
    assert AdapterDataSource.new_data == False
    if pre_call_index == 0:
        assert AdapterDataSource.double_buffer_read_index == 1
        assert result == AdapterDataSource.currentData[1]['data']
    elif pre_call_index == 1:
        assert AdapterDataSource.double_buffer_read_index == 0
        assert result == AdapterDataSource.currentData[0]['data']
    else:
        assert False

# tests for AdapterDataSource.has_more
def test_sbn_adapter_AdapterDataSource_has_more_returns_true(setup_teardown):
    # Arrange
    # Renew AdapterDataSource to ensure test independence
    AdapterDataSource = sbn_adapter.AdapterDataSource
    cut = AdapterDataSource.__new__(AdapterDataSource)

    # Act
    result = cut.has_more()

    # Assert
    assert result == True