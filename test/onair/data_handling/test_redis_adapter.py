# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"
import pytest
from unittest.mock import MagicMock

import onair.data_handling.redis_adapter as redis_adapter
from onair.data_handling.redis_adapter import DataSource
from onair.data_handling.on_air_data_source import OnAirDataSource
from onair.data_handling.on_air_data_source import ConfigKeyError

import redis
import threading

# __init__ tests
def test_redis_adapter_DataSource__init__sets_redis_values_then_connects(mocker):
def test_redis_adapter_DataSource__init__sets_redis_values_then_connects(mocker):
    # Arrange
    expected_server = []
    expected_server = []
    expected_subscriptions = MagicMock()

    arg_data_file = MagicMock()
    arg_meta_file = MagicMock()
    arg_ss_breakdown = MagicMock()

    fake_new_data_lock = MagicMock()

    cut = DataSource.__new__(DataSource)
    fake_order = MagicMock()
    fake_order.__len__.return_value = \
        pytest.gen.randint(1, 10) # from 1 to 10 arbitrary
    cut.order = fake_order

    mocker.patch.object(OnAirDataSource, '__init__', new=MagicMock())
    mocker.patch('threading.Lock', return_value=fake_new_data_lock)
    mocker.patch.object(cut, 'connect')

    # Act
    cut.__init__(arg_data_file, arg_meta_file, arg_ss_breakdown)

    # Assert
    assert OnAirDataSource.__init__.call_count == 1
    assert OnAirDataSource.__init__.call_args_list[0].args == (arg_data_file, arg_meta_file, arg_ss_breakdown)
    assert cut.servers == expected_server
    assert cut.servers == expected_server
    assert cut.new_data_lock == fake_new_data_lock
    assert threading.Lock.call_count == 1
    assert threading.Lock.call_count == 1
    assert cut.new_data == False
    assert cut.currentData == [{'headers':fake_order,
                                'data':list('-' * len(fake_order))},
                               {'headers':fake_order,
                                'data':list('-' * len(fake_order))}]
    assert cut.double_buffer_read_index == 0
    assert cut.connect.call_count == 1
    assert cut.connect.call_args_list[0].args == ()

# connect tests
def test_redis_adapter_DataSource_connect_establishes_server_with_initialized_attributes(mocker):
    # Arrange
    fake_server_configs = [{"address": MagicMock(), "port": 1234,"db": 1, "password": 'test', "subscriptions": ["state_0", "state_1"]}, {"address": '000.000.000.222', "port": 5678, "db": 2, "password": 'test2', "subscriptions" : ["state_2", "state_3"]}]

    fake_server = MagicMock()

    fake_message_listener = MagicMock()
    fake_listen_thread = MagicMock()

    fake_message_listener = MagicMock()
    fake_listen_thread = MagicMock()

    cut = DataSource.__new__(DataSource)
    cut.server_configs = fake_server_configs
    cut.servers = []
    cut.message_listener = fake_message_listener
    
    
    cut.server_configs = fake_server_configs
    cut.servers = []
    cut.message_listener = fake_message_listener
    
    
    mocker.patch(redis_adapter.__name__ + '.print_msg')
    mocker.patch('redis.Redis', return_value=fake_server)
    mocker.patch.object(fake_server, 'ping')
    mocker.patch('threading.Thread', return_value=fake_listen_thread)
    mocker.patch.object(fake_listen_thread, 'start')
    mocker.patch.object(fake_server, 'ping')
    mocker.patch('threading.Thread', return_value=fake_listen_thread)
    mocker.patch.object(fake_listen_thread, 'start')

    # Act
    cut.connect()

    # Assert
    assert redis_adapter.print_msg.call_count == 7
    assert redis_adapter.print_msg.call_count == 7
    assert redis_adapter.print_msg.call_args_list[0].args == ('Redis adapter connecting to server...',)
    assert redis_adapter.print_msg.call_args_list[1].args == ('... connected to server # 0!',)
    assert redis_adapter.print_msg.call_args_list[2].args == ('Subscribing to channel: state_0 on server # 0',) 
    assert redis_adapter.print_msg.call_args_list[3].args == ('Subscribing to channel: state_1 on server # 0',) 
    assert redis_adapter.print_msg.call_args_list[4].args == ('... connected to server # 1!',)
    assert redis_adapter.print_msg.call_args_list[5].args == ('Subscribing to channel: state_2 on server # 1',)
    assert redis_adapter.print_msg.call_args_list[6].args == ('Subscribing to channel: state_3 on server # 1',) 
     
    assert redis.Redis.call_count == 2
    assert redis.Redis.call_args_list[0].args == (fake_server_configs[0]["address"], fake_server_configs[0]["port"], fake_server_configs[0]["db"], fake_server_configs[0]["password"] )
    assert redis.Redis.call_args_list[1].args == (fake_server_configs[1]["address"], fake_server_configs[1]["port"], fake_server_configs[1]["db"], fake_server_configs[1]["password"] )
    
    assert fake_server.ping.call_count == 2
    assert cut.servers == [fake_server, fake_server]

# connect tests
def test_redis_adapter_DataSource_connect_establishes_server_with_default_attributes(mocker):
    # Arrange
    expected_address = 'localhost'
    expected_port = 6379
    expected_db = 0
    expected_password = ''

    fake_server_configs = [{"subscriptions": ["state_0", "state_1"]}, {"subscriptions" : ["state_2", "state_3"]}]

    fake_server = MagicMock()

    fake_message_listener = MagicMock()
    fake_listen_thread = MagicMock()

    cut = DataSource.__new__(DataSource)
    cut.server_configs = fake_server_configs
    cut.servers = []
    cut.message_listener = fake_message_listener
    
    
    mocker.patch(redis_adapter.__name__ + '.print_msg')
    mocker.patch('redis.Redis', return_value=fake_server)
    mocker.patch.object(fake_server, 'ping')
    mocker.patch('threading.Thread', return_value=fake_listen_thread)
    mocker.patch.object(fake_listen_thread, 'start')

    # Act
    cut.connect()

    # Assert
    assert redis.Redis.call_count == 2
    assert redis.Redis.call_args_list[0].args == (expected_address, expected_port, expected_db, expected_password )
    assert redis.Redis.call_args_list[1].args == (expected_address, expected_port, expected_db, expected_password )
    

def test_redis_adapter_DataSource_fails_to_connect_to_server_with_ping_and_states_no_subscriptions_(mocker):    
    fake_server_configs = [{"subscriptions": ["state_0", "state_1"]}, {"subscriptions": ["state_2", "state_3"]}]
    fake_server = MagicMock()

    cut = DataSource.__new__(DataSource)
    cut.server_configs = fake_server_configs
    cut.servers = []

    mocker.patch(redis_adapter.__name__ + '.print_msg')
    mocker.patch('redis.Redis', return_value=fake_server)
    mocker.patch.object(fake_server, 'ping', side_effect=ConnectionError)

    # Act
    cut.connect()   

    # Assert
    assert redis_adapter.print_msg.call_count == 3
    assert redis_adapter.print_msg.call_args_list[0].args == ("Redis adapter connecting to server...",)
    assert redis.Redis.call_count == 2
    assert fake_server.ping.call_count == 2
    assert cut.servers  == [fake_server, fake_server]
    
    assert redis_adapter.print_msg.call_args_list[0].args == ('Redis adapter connecting to server...',)
    assert redis_adapter.print_msg.call_args_list[1].args == ('Did not connect to server # 0. Not setting up subscriptions.', 'RED')
    assert redis_adapter.print_msg.call_args_list[2].args == ('Did not connect to server # 1. Not setting up subscriptions.', 'RED')
       

# subscribe_message tests
def test_redis_adapter_DataSource_subscribe_subscribes_to_each_given_subscription_and_starts_listening_when_server_available(mocker):
    # Arrange
    
    
    fake_server = MagicMock()
    fake_pubsub = MagicMock()
    fake_message_listener = MagicMock()
    fake_listen_thread = MagicMock()

    fake_server_configs = [{"subscriptions": ["state_0", "state_1"]}, {"subscriptions": ["state_2", "state_3"]}]
    fake_message_listener = MagicMock()
    fake_listen_thread = MagicMock()

    fake_server_configs = [{"subscriptions": ["state_0", "state_1"]}, {"subscriptions": ["state_2", "state_3"]}]

    cut = DataSource.__new__(DataSource)
    cut.server_configs = fake_server_configs
    cut.message_listener = fake_message_listener
    cut.servers = [{"subscriptions": ["state_0", "state_1"]}, {"subscriptions": ["state_2", "state_3"]}]
    cut.server_configs = fake_server_configs
    cut.message_listener = fake_message_listener
    cut.servers = [{"subscriptions": ["state_0", "state_1"]}, {"subscriptions": ["state_2", "state_3"]}]

    mocker.patch('redis.Redis', return_value=fake_server)
    mocker.patch('redis.Redis', return_value=fake_server)
    mocker.patch.object(fake_server, 'ping', return_value=True)
    mocker.patch.object(fake_server, 'pubsub', return_value=fake_pubsub)
    mocker.patch.object(fake_pubsub, 'subscribe')
    mocker.patch(redis_adapter.__name__ + '.print_msg')
    mocker.patch('threading.Thread', return_value=fake_listen_thread)
    mocker.patch.object(fake_listen_thread, 'start')
    mocker.patch('threading.Thread', return_value=fake_listen_thread)
    mocker.patch.object(fake_listen_thread, 'start')

    # Act
    cut.connect()
    cut.connect()

    # Assert
    assert fake_server.ping.call_count == 2
    assert fake_server.pubsub.call_count == 2

    #This is already checked in the first test. Should it be removed? Should the first test not check the subscription messages?
    assert redis_adapter.print_msg.call_count == 7
    assert redis_adapter.print_msg.call_args_list[0].args == ('Redis adapter connecting to server...',)
    assert redis_adapter.print_msg.call_args_list[1].args == ('... connected to server # 0!',)
    assert redis_adapter.print_msg.call_args_list[2].args == ('Subscribing to channel: state_0 on server # 0',) 
    assert redis_adapter.print_msg.call_args_list[3].args == ('Subscribing to channel: state_1 on server # 0',) 
    assert redis_adapter.print_msg.call_args_list[4].args == ('... connected to server # 1!',)
    assert redis_adapter.print_msg.call_args_list[5].args == ('Subscribing to channel: state_2 on server # 1',)
    assert redis_adapter.print_msg.call_args_list[6].args == ('Subscribing to channel: state_3 on server # 1',)     

    assert fake_pubsub.subscribe.call_count == 2
   
    assert threading.Thread.call_count == 2
    assert threading.Thread.call_args_list[0].kwargs == ({'target': cut.message_listener, 'args': (fake_pubsub,)})
    assert fake_listen_thread.start.call_count == 2

def test_redis_adapter_DataSource_subscribe_states_no_subscriptions_given_when_empty(mocker):
    # Arrange
    fake_servers = MagicMock()
    fake_servers = MagicMock()
    initial_pubsub = MagicMock()
    fake_message_listener = MagicMock()
    fake_listen_thread = MagicMock()

    fake_server_configs = [{'subscriptions': {}}]

    cut = DataSource.__new__(DataSource)
    cut.server_configs = fake_server_configs
    cut.message_listener = fake_message_listener
    cut.servers = []


    mocker.patch('redis.Redis', return_value=fake_servers)
    mocker.patch.object(fake_servers, 'ping', return_value=True)
    mocker.patch.object(fake_servers, 'pubsub', return_value=initial_pubsub)
    mocker.patch.object(initial_pubsub, 'subscribe')
    mocker.patch(redis_adapter.__name__ + '.print_msg')
    mocker.patch('threading.Thread', return_value=fake_listen_thread)
    mocker.patch.object(fake_listen_thread, 'start')


    # Act
    cut.connect()

    # Assert
    assert fake_servers.ping.call_count == 0
    assert fake_servers.pubsub.call_count == 0
    assert fake_servers.pubsub.subscribe.call_count == 0
    assert threading.Thread.call_count == 0
    assert fake_listen_thread.start.call_count == 0
    assert redis_adapter.print_msg.call_args_list[1].args == ("No subscriptions given! Redis server not created",)


# Note the self.server.ping during runtime will error, not actually return False, but that means code will never run
# this unit test is for completeness of coverage
# def test_redis_adapter_DataSource_subscribe_states_no_subscriptions_given_when_server_does_not_respond_to_ping(mocker):
#     # Arrange
#     fake_server = MagicMock()
#     initial_pubsub = MagicMock()
#     fake_subscription = MagicMock()
#     fake_listen_thread = MagicMock()
#     fake_message_listener = MagicMock()
#     fake_server_configs = [{"subscriptions": ["state_0", "state_1"]}, {"subscriptions": ["state_2", "state_3"]}]

#     cut = DataSource.__new__(DataSource)
#     cut.servers = fake_server
#     cut.pubsub = initial_pubsub
    

#     mocker.patch.object(fake_server, 'ping', return_value=False)
#     mocker.patch(redis_adapter.__name__ + '.print_msg')
#     mocker.patch.object(fake_server, 'pubsub')
#     mocker.patch('threading.Thread')
#     mocker.patch.object(fake_thread, 'start')

#     # Act
#     cut.connect()

#     # Assert
#     assert fake_server.ping.call_count == 1
#     assert fake_server.pubsub.call_count == 0
#     assert threading.Thread.call_count == 0
#     assert fake_thread.start.call_count == 0
#     assert cut.pubsub == initial_pubsub
#     assert redis_adapter.print_msg.call_args_list[1].args == ("No subscriptions given!",)

# # get_next tests
# def test_redis_adapter_DataSource_get_next_returns_expected_data_when_new_data_is_true_and_double_buffer_read_index_is_0():
#     # Arrange
#     # Renew DataSource to ensure test independence
#     cut = DataSource.__new__(DataSource)
#     cut.new_data = True
#     cut.new_data_lock = MagicMock()
#     cut.double_buffer_read_index = 0
#     pre_call_index = cut.double_buffer_read_index
#     expected_result = MagicMock()
#     cut.currentData = []
#     cut.currentData.append({'data': MagicMock()})
#     cut.currentData.append({'data': expected_result})

#     # Act
#     result = cut.get_next()

#     # Assert
#     assert cut.new_data == False
#     assert cut.double_buffer_read_index == 1
#     assert result == expected_result

def test_redis_adapter_DataSource_get_next_returns_expected_data_when_new_data_is_true_and_double_buffer_read_index_is_1():
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

def test_redis_adapter_DataSource_get_next_when_called_multiple_times_when_new_data_is_true():
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

def test_redis_adapter_DataSource_get_next_waits_until_data_is_available(mocker):
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
    mocker.patch(redis_adapter.__name__ + '.time.sleep')

    # Act
    result = cut.get_next()

    # Assert
    assert cut.has_data.call_count == num_falses + 1
    assert redis_adapter.time.sleep.call_count == num_falses
    assert cut.new_data == False
    if pre_call_index == 0:
        assert cut.double_buffer_read_index == 1
    elif pre_call_index == 1:
        assert cut.double_buffer_read_index == 0
    else:
        assert False

    assert result == expected_result

# has_more tests
def test_redis_adapter_DataSource_has_more_always_returns_True():
    cut = DataSource.__new__(DataSource)
    assert cut.has_more() == True

# message_listener tests
def test_redis_adapter_DataSource_message_listener_warns_of_exit_and_does_not_run_for_loop_when_listen_returns_StopIteration(mocker):
    # Arrange
    cut = DataSource.__new__(DataSource)

    fake_server = MagicMock()
    fake_server.fake_pubsub = MagicMock()
    fake_server = MagicMock()
    fake_server.fake_pubsub = MagicMock()
    fake_listener = MagicMock(name='fake_listener')
    fake_listener.__next__.side_effect = StopIteration
    mocker.patch.object(fake_server.fake_pubsub, 'listen', side_effect=[fake_listener])
    mocker.patch.object(fake_server.fake_pubsub, 'listen', side_effect=[fake_listener])
    mocker.patch(redis_adapter.__name__ + '.json.loads')
    mocker.patch(redis_adapter.__name__ + '.print_msg')

    # Act
    cut.message_listener(fake_server.fake_pubsub)
    cut.message_listener(fake_server.fake_pubsub)

    # Assert
    assert redis_adapter.json.loads.call_count == 0
    assert redis_adapter.print_msg.call_count == 1
    assert redis_adapter.print_msg.call_args_list[0].args == ("Redis subscription listener exited.", ['WARNING'])

def test_redis_adapter_DataSource_message_listener_prints_warning_when_receiving_non_message_type(mocker):
    # Arrange
    cut = DataSource.__new__(DataSource)

    #cut.pubsub = MagicMock()
    fake_server = MagicMock()
    fake_server.fake_pubsub = MagicMock()
    #cut.pubsub = MagicMock()
    fake_server = MagicMock()
    fake_server.fake_pubsub = MagicMock()
    ignored_message_types = ['subscribe', 'unsubscribe', 'psubscribe', 'punsubscribe', 'pmessage']
    fake_message = {}
    fake_message['type'] = pytest.gen.choice(ignored_message_types)
    fake_message['channel'] = str(MagicMock(name='fake_message')).encode('utf-8')
    mocker.patch.object(fake_server.fake_pubsub, 'listen', return_value=[fake_message])
    mocker.patch.object(fake_server.fake_pubsub, 'listen', return_value=[fake_message])
    mocker.patch(redis_adapter.__name__ + '.json.loads')
    mocker.patch(redis_adapter.__name__ + '.print_msg')

    # Act
    cut.message_listener(fake_server.fake_pubsub)
    cut.message_listener(fake_server.fake_pubsub)

    # Assert
    assert redis_adapter.json.loads.call_count == 0
    assert redis_adapter.print_msg.call_count == 2
    assert redis_adapter.print_msg.call_args_list[0].args == (
        f"Redis adapter: channel '{fake_message['channel'].decode()}' received " \
                           f"message type: {fake_message['type']}.", ['WARNING'])
    assert redis_adapter.print_msg.call_args_list[1].args == (
        "Redis subscription listener exited.", ['WARNING'])

def test_redis_adapter_DataSource_message_listener_prints_warning_when_data_not_json_format_and_does_not_update_frame(mocker):
    # Arrange
    cut = DataSource.__new__(DataSource)
    fake_server = MagicMock()
    fake_server.fake_pubsub = MagicMock()
    fake_server = MagicMock()
    fake_server.fake_pubsub = MagicMock()
    fake_message = {}
    fake_message['type'] = 'message'
    fake_message['channel'] = str(
        MagicMock(name='fake_message_channel')).encode('utf-8')
    fake_message['data'] = str(MagicMock(name='fake_message_data'))
    mocker.patch.object(fake_server.fake_pubsub, 'listen', return_value=[fake_message])
    mocker.patch.object(fake_server.fake_pubsub, 'listen', return_value=[fake_message])
    mocker.patch(redis_adapter.__name__ + '.json.loads', side_effect=ValueError)
    mocker.patch(redis_adapter.__name__ + '.print_msg')

    # Act
    cut.message_listener(fake_server.fake_pubsub)
    cut.message_listener(fake_server.fake_pubsub)

    # Assert
    assert redis_adapter.json.loads.call_count == 1
    assert redis_adapter.json.loads.call_args_list[0].args == (
        fake_message['data'], )
    assert redis_adapter.print_msg.call_count == 2
    assert redis_adapter.print_msg.call_args_list[0].args == (
        f'Subscribed channel `{fake_message["channel"].decode()}\' message ' \
         'received but is not in json format.\nMessage:\n' \
        f'{fake_message["data"]}', ['WARNING'])
    assert redis_adapter.print_msg.call_args_list[1].args == (
        "Redis subscription listener exited.", ['WARNING'])

def test_redis_adapter_DataSource_message_listener_warns_user_when_processed_data_did_not_contain_time(mocker):
    # Arrange
    cut = DataSource.__new__(DataSource)
    cut.double_buffer_read_index = pytest.gen.choice([0 , 1])
    cut.currentData = {0: {'headers': [], 'data': []},
                       1: {'headers': [], 'data': []}}
    fake_server = MagicMock()
    fake_server.fake_pubsub = MagicMock()
    fake_server = MagicMock()
    fake_server.fake_pubsub = MagicMock()
    cut.new_data_lock = MagicMock()
    cut.new_data = False

    fake_message = {}
    fake_message['type'] = 'message'
    fake_message['channel'] = str(
        MagicMock(name='fake_message_channel')).encode('utf-8')
    fake_message['data'] = '{}' # empty_message
    mocker.patch.object(fake_server.fake_pubsub, 'listen', return_value=[fake_message])
    mocker.patch.object(fake_server.fake_pubsub, 'listen', return_value=[fake_message])
    mocker.patch(redis_adapter.__name__ + '.json.loads', return_value={})
    mocker.patch(redis_adapter.__name__ + '.print_msg')

    # Act
    cut.message_listener(fake_server.fake_pubsub)
    cut.message_listener(fake_server.fake_pubsub)

    # Assert
    assert redis_adapter.json.loads.call_count == 1
    assert redis_adapter.json.loads.call_args_list[0].args == (
        fake_message['data'], )
    assert redis_adapter.print_msg.call_count == 2
    assert redis_adapter.print_msg.call_args_list[0].args == (
        f'Message from channel `{fake_message["channel"].decode()}\' ' \
        f'did not contain `time\' key\nMessage:\n{fake_message["data"]}', \
         ['WARNING'])
    assert redis_adapter.print_msg.call_args_list[1].args == (
        "Redis subscription listener exited.", ['WARNING'])

def test_redis_adapter_DataSource_message_listener_warns_of_received_key_that_does_not_exist_in_header(mocker):
    # Arrange
    cut = DataSource.__new__(DataSource)
    cut.double_buffer_read_index = pytest.gen.choice([0 , 1])
    cut.currentData = {0: {'headers': ['time'],
                           'data': ['-']},
                       1: {'headers': ['time'],
                           'data': ['-']}}
    fake_server = MagicMock()
    fake_server.fake_pubsub = MagicMock()
    fake_server = MagicMock()
    fake_server.fake_pubsub = MagicMock()
    cut.new_data_lock = MagicMock()
    cut.new_data = False

    fake_message = {}
    fake_message['type'] = 'message'
    fake_message['channel'] = str(
        MagicMock(name='fake_message_channel')).encode('utf-8')
    fake_message['data'] = '{"time":0, "unknown_key":0}'
    mocker.patch.object(fake_server.fake_pubsub, 'listen', return_value=[fake_message])
    mocker.patch.object(fake_server.fake_pubsub, 'listen', return_value=[fake_message])
    mocker.patch(redis_adapter.__name__ + '.json.loads', return_value={"time":0, "unknown_key":0})
    mocker.patch(redis_adapter.__name__ + '.print_msg')

    # Act
    cut.message_listener(fake_server.fake_pubsub)
    cut.message_listener(fake_server.fake_pubsub)

    # Assert
    assert redis_adapter.json.loads.call_count == 1
    assert redis_adapter.json.loads.call_args_list[0].args == (
        fake_message['data'], )
    assert redis_adapter.print_msg.call_count == 2
    assert redis_adapter.print_msg.call_args_list[0].args == (
         f"Unused key `unknown_key' in message " \
         f'from channel `{fake_message["channel"].decode()}.\'', ['WARNING'])
    assert redis_adapter.print_msg.call_args_list[1].args == (
        "Redis subscription listener exited.", ['WARNING'])

def test_redis_adapter_DataSource_message_listener_warns_of_expected_keys_that_do_not_appear_in_message(mocker):
    # Arrange
    cut = DataSource.__new__(DataSource)
    cut.double_buffer_read_index = pytest.gen.choice([0 , 1])  
    fake_server = MagicMock()
    fake_server.fake_pubsub = MagicMock()
    cut.double_buffer_read_index = pytest.gen.choice([0 , 1])  
    fake_server = MagicMock()
    fake_server.fake_pubsub = MagicMock()
    cut.new_data_lock = MagicMock()
    cut.new_data = False

    fake_message = {}
    fake_message['type'] = 'message'
    fake_message['channel'] = str(
        MagicMock(name='fake_message_channel')).encode('utf-8')
    cut.currentData = {0: {'headers': ['time',
                                      f'{fake_message["channel"].decode()}' \
                                       '.missing_key'],
                           'data': ['-', '-']},
                       1: {'headers': ['time',
                                      f'{fake_message["channel"].decode()}' \
                                       '.missing_key'],
                           'data': ['-', '-']}}
    fake_message['data'] = '{}'
    mocker.patch.object(fake_server.fake_pubsub, 'listen', return_value=[fake_message])
    mocker.patch.object(fake_server.fake_pubsub, 'listen', return_value=[fake_message])
    mocker.patch(redis_adapter.__name__ + '.json.loads', return_value={})
    mocker.patch(redis_adapter.__name__ + '.print_msg')

    # Act
    cut.message_listener(fake_server.fake_pubsub)
    cut.message_listener(fake_server.fake_pubsub)

    # Assert
    assert redis_adapter.json.loads.call_count == 1
    assert redis_adapter.json.loads.call_args_list[0].args == (
        fake_message['data'], )
    assert redis_adapter.print_msg.call_count == 3
    assert redis_adapter.print_msg.call_args_list[0].args == (
        f'Message from channel `{fake_message["channel"].decode()}\' ' \
        f'did not contain `{fake_message["channel"].decode()}.missing_key\'' \
        f' key\nMessage:\n{fake_message["data"]}', \
         ['WARNING'])
    assert redis_adapter.print_msg.call_args_list[1].args == (
        f'Message from channel `{fake_message["channel"].decode()}\' ' \
        f'did not contain `time\' key\nMessage:\n{fake_message["data"]}', \
         ['WARNING'])
    assert redis_adapter.print_msg.call_args_list[2].args == (
        "Redis subscription listener exited.", ['WARNING'])

def test_redis_adapter_DataSource_message_listener_updates_new_data_with_received_data_by_channel_and_key_matched_to_frame_header(mocker):
    # Arrange
    cut = DataSource.__new__(DataSource)
    cut.double_buffer_read_index = pytest.gen.choice([0 , 1])
    fake_server = MagicMock()
    fake_server.fake_pubsub = MagicMock()
    fake_server = MagicMock()
    fake_server.fake_pubsub = MagicMock()
    cut.new_data_lock = MagicMock()
    cut.new_data = False

    fake_message = {}
    fake_message['type'] = 'message'
    fake_message['channel'] = str(
        MagicMock(name='fake_message_channel')).encode('utf-8')
    cut.currentData = {0: {'headers': ['time',
                                      f'{fake_message["channel"].decode()}' \
                                       '.correct_key', 'fakeotherchannel.x'],
                           'data': ['-', '-', '0']},
                       1: {'headers': ['time',
                                      f'{fake_message["channel"].decode()}' \
                                       '.correct_key', 'fakeotherchannel.x'],
                           'data': ['-', '-', '0']}}
    fake_message['data'] = '{}'
    mocker.patch.object(fake_server.fake_pubsub, 'listen', return_value=[fake_message])
    mocker.patch.object(fake_server.fake_pubsub, 'listen', return_value=[fake_message])
    fake_data = {
        'time': pytest.gen.randint(1, 100), # from 1 to 100 arbitrary
        'correct_key': pytest.gen.randint(1, 100), # from 1 to 100 arbitrary
    }
    mocker.patch(redis_adapter.__name__ + '.json.loads',
                 return_value=fake_data)
    mocker.patch(redis_adapter.__name__ + '.print_msg')

    # Act
    cut.message_listener(fake_server.fake_pubsub)
    cut.message_listener(fake_server.fake_pubsub)

    # Assert
    assert redis_adapter.json.loads.call_count == 1
    assert redis_adapter.json.loads.call_args_list[0].args == (
        fake_message['data'], )
    assert cut.new_data == True
    print(cut.currentData[cut.double_buffer_read_index])
    assert cut.currentData[(cut.double_buffer_read_index + 1) % 2]['data'] == \
        [fake_data['time'], fake_data['correct_key'], '-']
    assert redis_adapter.print_msg.call_count == 1
    assert redis_adapter.print_msg.call_args_list[0].args == (
        "Redis subscription listener exited.", ['WARNING'])

# has_data tests
def test_redis_adapter_DataSource_has_data_returns_instance_new_data():
    cut = DataSource.__new__(DataSource)
    expected_result = MagicMock()
    cut.new_data = expected_result

    result = cut.has_data()

    assert result == expected_result

# redis_adapter parse_meta_data tests
def test_redis_adapter_DataSource_parse_meta_data_file_raises_ConfigKeyError_when_order_is_not_in_config_file(mocker):
    # Arrange
    cut = DataSource.__new__(DataSource)
    arg_configFile = MagicMock()
    arg_ss_breakdown = MagicMock()

    expected_extracted_configs = MagicMock()
    expected_subscriptions = [MagicMock()] * pytest.gen.randint(0, 10) # 0 to 10 arbitrary
    fake_meta = {'fake_other_stuff': MagicMock(),
                 'redis_subscriptions':expected_subscriptions}

    mocker.patch(redis_adapter.__name__ + '.extract_meta_data_handle_ss_breakdown', return_value=expected_extracted_configs)
    mocker.patch(redis_adapter.__name__ + '.parseJson', return_value=fake_meta)

    exception_message = (f'Config file: \'{arg_configFile}\' ' \
                          'missing required key \'order\'')

    # Act
    with pytest.raises(ConfigKeyError) as e_info:
        cut.parse_meta_data_file(arg_configFile, arg_ss_breakdown, )

    # Assert
    assert redis_adapter.extract_meta_data_handle_ss_breakdown.call_count == 1
    assert redis_adapter.extract_meta_data_handle_ss_breakdown.call_args_list[0].args == (arg_configFile, arg_ss_breakdown)
    assert redis_adapter.parseJson.call_count == 1
    assert redis_adapter.parseJson.call_args_list[0].args == (arg_configFile, )
    assert e_info.match(exception_message)

def test_redis_adapter_DataSource_parse_meta_data_file_returns_call_to_extract_meta_data_handle_ss_breakdown(mocker):
def test_redis_adapter_DataSource_parse_meta_data_file_returns_call_to_extract_meta_data_handle_ss_breakdown(mocker):
    # Arrange
    cut = DataSource.__new__(DataSource)
    arg_configFile = MagicMock()
    arg_ss_breakdown = MagicMock()

    expected_extracted_configs = MagicMock()
    fake_meta = {'fake_other_stuff': MagicMock(),
                 'order': MagicMock()}
                 'order': MagicMock()}

    mocker.patch(redis_adapter.__name__ + '.extract_meta_data_handle_ss_breakdown', return_value=expected_extracted_configs)
    mocker.patch(redis_adapter.__name__ + '.parseJson', return_value=fake_meta)

    # Act
    result = cut.parse_meta_data_file(arg_configFile, arg_ss_breakdown, )

    # Assert
    assert redis_adapter.extract_meta_data_handle_ss_breakdown.call_count == 1
    assert redis_adapter.extract_meta_data_handle_ss_breakdown.call_args_list[0].args == (arg_configFile, arg_ss_breakdown)
    assert redis_adapter.parseJson.call_count == 1
    assert redis_adapter.parseJson.call_args_list[0].args == (arg_configFile, )
    assert result == expected_extracted_configs

# def test_redis_adapter_DataSource_parse_meta_data_file_returns_call_to_extract_meta_data_for_redis_server_configurations(mocker):
#     # Arrange
#     cut = DataSource.__new__(DataSource)
#     arg_configFile = MagicMock()
#     arg_ss_breakdown = MagicMock()


#     expected_extracted_configs = MagicMock()
#     expected_subscriptions = [MagicMock()] * pytest.gen.randint(0, 10) # 0 to 10 arbitrary
#     expected_address = MagicMock()
#     expected_port = MagicMock()
#     expected_password = MagicMock()
#     expected_redis_configs = {'address': expected_address, 'port': expected_port, 'password': expected_password}
#     fake_meta = {'fake_other_stuff': MagicMock(),
#                  'order': MagicMock(),
#                  'redis_subscriptions': expected_subscriptions,
#                  'redis': expected_redis_configs}

#     mocker.patch(redis_adapter.__name__ + '.extract_meta_data_handle_ss_breakdown', return_value=expected_extracted_configs)
#     mocker.patch(redis_adapter.__name__ + '.parseJson', return_value=fake_meta)

#     # Act
#     result = cut.parse_meta_data_file(arg_configFile, arg_ss_breakdown, )

#     # Assert
#     assert redis_adapter.extract_meta_data_handle_ss_breakdown.call_count == 1
#     assert redis_adapter.extract_meta_data_handle_ss_breakdown.call_args_list[0].args == (arg_configFile, arg_ss_breakdown)
#     assert redis_adapter.parseJson.call_count == 1
#     assert redis_adapter.parseJson.call_args_list[0].args == (arg_configFile, )
#     assert cut.subscriptions == expected_subscriptions
#     assert result == expected_extracted_configs
#     assert cut.address == expected_address
#     assert cut.port == expected_port
#     assert cut.password == expected_password

# def test_redis_adapter_DataSource_parse_meta_data_file_returns_call_to_extract_meta_data_handle_ss_breakdown_and_sets_subscriptions_to_empty_when_none_given(mocker):
#     # Arrange
#     cut = DataSource.__new__(DataSource)
#     arg_configFile = MagicMock()
#     arg_ss_breakdown = MagicMock()

#     fake_configs = {'fake_other_stuff': MagicMock()}
#     fake_meta = {'order': MagicMock()}

#     mocker.patch(redis_adapter.__name__ + '.extract_meta_data_handle_ss_breakdown', return_value=fake_configs)
#     mocker.patch(redis_adapter.__name__ + '.parseJson', return_value=fake_meta)

#     # Act
#     result = cut.parse_meta_data_file(arg_configFile, arg_ss_breakdown, )

#     # Assert
#     assert redis_adapter.extract_meta_data_handle_ss_breakdown.call_count == 1
#     assert redis_adapter.extract_meta_data_handle_ss_breakdown.call_args_list[0].args == (arg_configFile, arg_ss_breakdown)
#     assert redis_adapter.parseJson.call_count == 1
#     assert redis_adapter.parseJson.call_args_list[0].args == (arg_configFile, )
#     assert cut.subscriptions == []
#     assert result == fake_configs

# redis_adapter get_vehicle_metadata tests
def test_redis_adapter_DataSource_get_vehicle_metadata_returns_list_of_headers_and_list_of_test_assignments():
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

# redis_adapter process_data_file tests
def test_redis_adapter_DataSource_process_data_file_does_nothing():
    # Arrange
    cut = DataSource.__new__(DataSource)
    arg_data_file = MagicMock()

    expected_result = None

    # Act
    result = cut.process_data_file(arg_data_file)

    # Assert
    assert result == expected_result


def test_redis_adapter_DataSource_parse_meta_data_file_redis_in_keys_subscriptions_exist_and_adds_redis_to_server_configs(mocker):
    # Arrange
    cut = DataSource.__new__(DataSource)
    arg_configFile = MagicMock()
    arg_ss_breakdown = MagicMock()

    fake_server_configs = [{"subscriptions": ["state_0", "state_1"]}, {"subscriptions": ["state_2", "state_3"]}]
    cut.server_configs = MagicMock()

    expected_extracted_configs = MagicMock()
    fake_meta = {'fake_other_stuff': MagicMock(),
                  'redis': fake_server_configs,
                  'order': MagicMock()}

    mocker.patch(redis_adapter.__name__ + '.extract_meta_data_handle_ss_breakdown', return_value=expected_extracted_configs)
    mocker.patch(redis_adapter.__name__ + '.parseJson', return_value=fake_meta)

    # Act
    result = cut.parse_meta_data_file(arg_configFile, arg_ss_breakdown, )

    # Assert
    assert redis_adapter.extract_meta_data_handle_ss_breakdown.call_count == 1
    assert redis_adapter.extract_meta_data_handle_ss_breakdown.call_args_list[0].args == (arg_configFile, arg_ss_breakdown)
    assert redis_adapter.parseJson.call_count == 1
    assert redis_adapter.parseJson.call_args_list[0].args == (arg_configFile, )
    assert result == expected_extracted_configs
    assert cut.server_configs == fake_meta['redis']

def test_redis_adapter_DataSource_parse_meta_data_file_redis_in_keys_subscriptions_do_not_exist(mocker):
    # Arrange
    cut = DataSource.__new__(DataSource)
    arg_configFile = MagicMock()
    arg_ss_breakdown = MagicMock()

    fake_server_configs = [{"address": 1}]
    cut.server_configs = MagicMock()

    expected_extracted_configs = MagicMock()
    fake_meta = {'fake_other_stuff': MagicMock(),
                  'redis': fake_server_configs,
                  'order': MagicMock()}

    mocker.patch(redis_adapter.__name__ + '.extract_meta_data_handle_ss_breakdown', return_value=expected_extracted_configs)
    mocker.patch(redis_adapter.__name__ + '.parseJson', return_value=fake_meta)

    # Act
    with pytest.raises(ConfigKeyError) as e_info:
        cut.parse_meta_data_file(arg_configFile, arg_ss_breakdown, )
    

    # Assert
    assert redis_adapter.extract_meta_data_handle_ss_breakdown.call_count == 1
    assert redis_adapter.extract_meta_data_handle_ss_breakdown.call_args_list[0].args == (arg_configFile, arg_ss_breakdown)
    assert redis_adapter.parseJson.call_count == 1
    assert redis_adapter.parseJson.call_args_list[0].args == (arg_configFile, )
    assert cut.server_configs == []
    assert e_info.match(f'Config file: \'{arg_configFile}\' missing required key \'subscriptions\' from 0 in key \'redis\'')  

def test_redis_adapter_DataSource_parse_meta_data_file_redis_in_keys_subscriptions_exist_and_adds_redis_to_server_configs(mocker):
    # Arrange
    cut = DataSource.__new__(DataSource)
    arg_configFile = MagicMock()
    arg_ss_breakdown = MagicMock()

    fake_server_configs = [{"subscriptions": ["state_0", "state_1"]}, {"subscriptions": ["state_2", "state_3"]}]
    cut.server_configs = MagicMock()

    expected_extracted_configs = MagicMock()
    fake_meta = {'fake_other_stuff': MagicMock(),
                  'redis': fake_server_configs,
                  'order': MagicMock()}

    mocker.patch(redis_adapter.__name__ + '.extract_meta_data_handle_ss_breakdown', return_value=expected_extracted_configs)
    mocker.patch(redis_adapter.__name__ + '.parseJson', return_value=fake_meta)

    # Act
    result = cut.parse_meta_data_file(arg_configFile, arg_ss_breakdown, )

    # Assert
    assert redis_adapter.extract_meta_data_handle_ss_breakdown.call_count == 1
    assert redis_adapter.extract_meta_data_handle_ss_breakdown.call_args_list[0].args == (arg_configFile, arg_ss_breakdown)
    assert redis_adapter.parseJson.call_count == 1
    assert redis_adapter.parseJson.call_args_list[0].args == (arg_configFile, )
    assert result == expected_extracted_configs
    assert cut.server_configs == fake_meta['redis']

def test_redis_adapter_DataSource_parse_meta_data_file_redis_in_keys_subscriptions_do_not_exist(mocker):
    # Arrange
    cut = DataSource.__new__(DataSource)
    arg_configFile = MagicMock()
    arg_ss_breakdown = MagicMock()

    fake_server_configs = [{"address": 1}]
    cut.server_configs = MagicMock()

    expected_extracted_configs = MagicMock()
    fake_meta = {'fake_other_stuff': MagicMock(),
                  'redis': fake_server_configs,
                  'order': MagicMock()}

    mocker.patch(redis_adapter.__name__ + '.extract_meta_data_handle_ss_breakdown', return_value=expected_extracted_configs)
    mocker.patch(redis_adapter.__name__ + '.parseJson', return_value=fake_meta)

    # Act
    with pytest.raises(ConfigKeyError) as e_info:
        cut.parse_meta_data_file(arg_configFile, arg_ss_breakdown, )
    

    # Assert
    assert redis_adapter.extract_meta_data_handle_ss_breakdown.call_count == 1
    assert redis_adapter.extract_meta_data_handle_ss_breakdown.call_args_list[0].args == (arg_configFile, arg_ss_breakdown)
    assert redis_adapter.parseJson.call_count == 1
    assert redis_adapter.parseJson.call_args_list[0].args == (arg_configFile, )
    assert cut.server_configs == []
    assert e_info.match(f'Config file: \'{arg_configFile}\' missing required key \'subscriptions\' from 0 in key \'redis\'')  