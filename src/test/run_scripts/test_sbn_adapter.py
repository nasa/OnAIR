import pytest
from mock import MagicMock, PropertyMock
import src.run_scripts.sbn_adapter as sbn_adapter
from src.run_scripts.sbn_adapter import AdapterDataSource

# add_time function is not tested because it can not be accessed outside of the message_listener_thread function

# tests for message_listener_thread
def test_sbn_adapter_message_listener_thread_(mocker):
    # Arrange

    # Act

    # Assert
    assert True

# ---------- Tests for AdapterDataSource class ---------

# tests for AdapterDataSource variables
def test_sbn_adapter_adapter_data_source_current_data_equals_expected_value():
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
            for field_name, field_type in data_struct._fields_[1:]:
                expected_current_data[x]['headers'].append('{}.{}.{}'.format(app_name, struct_name, str(field_name)))
            expected_current_data[x]['data'].extend([0]*len(data_struct._fields_[1:])) #initialize all the data arrays with zero

    # Act
    cut = AdapterDataSource.__new__(AdapterDataSource)

    # Assert
    assert cut.currentData == expected_current_data

# tests for AdapterDataSource.connect
def test_sbn_adapter_adapter_data_source_connect_when_msgID_lookup_table_has_zero_keys(mocker):
    # Arrange
    fake_listener_thread = MagicMock()
    fake_msgID_lookup_table = MagicMock()
    fake_msgID_lookup_table_keys = []

    mocker.patch('src.run_scripts.sbn_adapter.time.sleep')
    mocker.patch('src.run_scripts.sbn_adapter.os.chdir')
    mocker.patch('src.run_scripts.sbn_adapter.sbn.sbn_load_and_init')
    mocker.patch('src.run_scripts.sbn_adapter.message_listener_thread', fake_listener_thread)
    mocker.patch('src.run_scripts.sbn_adapter.threading.Thread', return_value=fake_listener_thread)
    mocker.patch.object(fake_listener_thread, 'start')
    mocker.patch('src.run_scripts.sbn_adapter.msgID_lookup_table', fake_msgID_lookup_table)
    mocker.patch.object(fake_msgID_lookup_table, 'keys', return_value=fake_msgID_lookup_table_keys)
    mocker.patch('src.run_scripts.sbn_adapter.sbn.subscribe')
    
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

def test_sbn_adapter_adapter_data_source_connect_when_msgID_lookup_table_has_one_key(mocker):
    # Arrange
    fake_listener_thread = MagicMock()
    fake_msgID_lookup_table = MagicMock()
    fake_msgID = MagicMock()
    fake_msgID_lookup_table_keys = [fake_msgID]

    mocker.patch('src.run_scripts.sbn_adapter.time.sleep')
    mocker.patch('src.run_scripts.sbn_adapter.os.chdir')
    mocker.patch('src.run_scripts.sbn_adapter.sbn.sbn_load_and_init')
    mocker.patch('src.run_scripts.sbn_adapter.message_listener_thread', fake_listener_thread)
    mocker.patch('src.run_scripts.sbn_adapter.threading.Thread', return_value=fake_listener_thread)
    mocker.patch.object(fake_listener_thread, 'start')
    mocker.patch('src.run_scripts.sbn_adapter.msgID_lookup_table', fake_msgID_lookup_table)
    mocker.patch.object(fake_msgID_lookup_table, 'keys', return_value=fake_msgID_lookup_table_keys)
    mocker.patch('src.run_scripts.sbn_adapter.sbn.subscribe')
    
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
    
def test_sbn_adapter_adapter_data_source_connect_when_msgID_lookup_table_has_multiple_keys(mocker):
    # Arrange
    fake_listener_thread = MagicMock()
    fake_msgID_lookup_table = MagicMock()
    num_keys = pytest.gen.randint(2, 10) # arbitrary, from 2 to 10
    fake_msgID_lookup_table_keys = []
    for i in range(num_keys):
        fake_msgID_lookup_table_keys.append(MagicMock())

    mocker.patch('src.run_scripts.sbn_adapter.time.sleep')
    mocker.patch('src.run_scripts.sbn_adapter.os.chdir')
    mocker.patch('src.run_scripts.sbn_adapter.sbn.sbn_load_and_init')
    mocker.patch('src.run_scripts.sbn_adapter.message_listener_thread', fake_listener_thread)
    mocker.patch('src.run_scripts.sbn_adapter.threading.Thread', return_value=fake_listener_thread)
    mocker.patch.object(fake_listener_thread, 'start')
    mocker.patch('src.run_scripts.sbn_adapter.msgID_lookup_table', fake_msgID_lookup_table)
    mocker.patch.object(fake_msgID_lookup_table, 'keys', return_value=fake_msgID_lookup_table_keys)
    mocker.patch('src.run_scripts.sbn_adapter.sbn.subscribe')
    
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
def test_sbn_adapter_adapter_data_source_subscribe_message_when_msgid_is_not_a_list(mocker):
    # Arrange
    arg_msgid = MagicMock()
    
    mocker.patch('src.run_scripts.sbn_adapter.sbn.subscribe')

    cut = AdapterDataSource.__new__(AdapterDataSource)

    # Act
    cut.subscribe_message(arg_msgid)

    # Assert
    assert sbn_adapter.sbn.subscribe.call_count == 1
    assert sbn_adapter.sbn.subscribe.call_args_list[0].args == (arg_msgid,)

def test_sbn_adapter_adapter_data_source_subscribe_message_when_msgid_is_an_empty_list(mocker):
    # Arrange
    arg_msgid = []
    
    mocker.patch('src.run_scripts.sbn_adapter.sbn.subscribe')

    cut = AdapterDataSource.__new__(AdapterDataSource)

    # Act
    cut.subscribe_message(arg_msgid)

    # Assert
    assert sbn_adapter.sbn.subscribe.call_count == 0

def test_sbn_adapter_adapter_data_source_subscribe_message_when_msgid_is_a_list_with_only_one_element(mocker):
    # Arrange
    arg_msgid = [MagicMock()]
    
    mocker.patch('src.run_scripts.sbn_adapter.sbn.subscribe')

    cut = AdapterDataSource.__new__(AdapterDataSource)

    # Act
    cut.subscribe_message(arg_msgid)

    # Assert
    assert sbn_adapter.sbn.subscribe.call_count == 1
    assert sbn_adapter.sbn.subscribe.call_args_list[0].args == (arg_msgid[0],)

def test_sbn_adapter_adapter_data_source_subscribe_message_when_msgid_is_a_list_of_multiple_elements(mocker):
    # Arrange
    list_length = pytest.gen.randint(2,10) # arbitrary, from 2 to 10
    arg_msgid = []
    for i in range(list_length):
        arg_msgid.append(MagicMock())
    
    mocker.patch('src.run_scripts.sbn_adapter.sbn.subscribe')

    cut = AdapterDataSource.__new__(AdapterDataSource)

    # Act
    cut.subscribe_message(arg_msgid)

    # Assert
    assert sbn_adapter.sbn.subscribe.call_count == list_length
    for i in range(list_length):
        assert sbn_adapter.sbn.subscribe.call_args_list[i].args == (arg_msgid[i],)
    
# tests for AdapterDataSource.get_next
def test_sbn_adapter_adapter_data_source_get_next_when_new_data_is_true():
    # Arrange
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

def test_sbn_adapter_adapter_data_source_get_next_when_called_multiple_times_when_new_data_is_true():
    # Arrange
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
    
def test_sbn_adapter_adapter_data_source_get_next_behavior_when_new_data_is_false_then_true(mocker):
    # Arrange
    cut = AdapterDataSource.__new__(AdapterDataSource)
    pre_call_index = AdapterDataSource.double_buffer_read_index

    num_falses = pytest.gen.randint(1, 10)
    side_effect_list = [False] * num_falses
    side_effect_list.append(True)

    AdapterDataSource.new_data = PropertyMock(side_effect=side_effect_list)
    mocker.patch('src.run_scripts.sbn_adapter.time.sleep')

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
def test_sbn_adapter_adapter_data_source_has_more_returns_true(mocker):
    # Arrange
    cut = AdapterDataSource.__new__(AdapterDataSource)

    # Act
    result = cut.has_more()

    # Assert
    assert result == True