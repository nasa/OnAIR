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
import onair.src.run_scripts.redis_adapter as redis_adapter
from onair.src.run_scripts.redis_adapter import AdapterDataSource
from importlib import reload
import redis
import threading

# __init__ tests
def test_redis_adapter_AdapterDataSource__init__sets_all_3_redis_arguments_for_later_use():
    # Arrange
    expected_address = 'localhost'
    expected_port = 6379
    expected_db = 0
    expected_server = None

    cut = AdapterDataSource.__new__(AdapterDataSource)

    # Act
    cut.__init__()

    # Assert
    assert cut.address == expected_address
    assert cut.port == expected_port
    assert cut.db == expected_db
    assert cut.server == expected_server

# connect tests
def test_redis_adapter_AdapterDataSource_connect_establishes_server_with_initialized_attributes(mocker):
    # Arrange
    expected_address = MagicMock()
    expected_port = MagicMock()
    expected_db = MagicMock()
    fake_server = MagicMock()
    
    cut = AdapterDataSource.__new__(AdapterDataSource)
    cut.address = expected_address
    cut.port = expected_port
    cut.db = expected_db

    mocker.patch('redis.Redis', return_value=fake_server)

    # Act
    cut.connect()

    # Assert
    assert redis.Redis.call_count == 1
    assert redis.Redis.call_args_list[0].args == (expected_address, expected_port, expected_db)
    assert cut.server == fake_server

# subscribe_message tests
def test_redis_adapter_AdapterDataSource_subscribe_message_and_thread_start_success_when_server_available(mocker):
    # Arrange
    arg_channel = str(MagicMock())
    fake_server = MagicMock()
    fake_pubsub = MagicMock()
    fake_subscription = MagicMock()
    fake_thread = MagicMock()
    cut = AdapterDataSource.__new__(AdapterDataSource)
    cut.server = fake_server

    mocker.patch.object(fake_server, 'ping', return_value=True)
    mocker.patch.object(fake_server, 'pubsub', return_value=fake_pubsub)
    mocker.patch.object(fake_pubsub, 'subscribe', return_value=fake_subscription)
    mocker.patch('threading.Thread', return_value=fake_thread)
    mocker.patch.object(fake_thread, 'start')

    # Act
    cut.subscribe_message(arg_channel)

    # Assert
    assert fake_server.ping.call_count == 1
    assert fake_server.pubsub.call_count == 1
    assert fake_pubsub.subscribe.call_count == 1
    assert fake_pubsub.subscribe.call_args_list[0].args == (arg_channel,)
    assert threading.Thread.call_count == 1
    assert threading.Thread.call_args_list[0].kwargs == ({'target': cut.message_listener})
    assert fake_thread.start.call_count == 1
    assert cut.pubsub == fake_pubsub

def test_redis_adapter_AdapterDataSource_subscribe_message_does_nothing_on_False(mocker):
    # Arrange
    arg_channel = str(MagicMock())
    fake_server = MagicMock()
    initial_pubsub = MagicMock()
    fake_subscription = MagicMock()
    fake_thread = MagicMock()
    cut = AdapterDataSource.__new__(AdapterDataSource)
    cut.server = fake_server
    cut.pubsub = initial_pubsub

    mocker.patch.object(fake_server, 'ping', return_value=False)
    mocker.patch.object(fake_server, 'pubsub')
    mocker.patch('threading.Thread')
    mocker.patch.object(fake_thread, 'start')

    # Act
    cut.subscribe_message(arg_channel)

    # Assert
    assert fake_server.ping.call_count == 1
    assert fake_server.pubsub.call_count == 0
    assert threading.Thread.call_count == 0
    assert fake_thread.start.call_count == 0
    assert cut.pubsub == initial_pubsub

# get_next tests

def test_redis_adapter_AdapterDataSource_get_next_when_new_data_is_true():
    # Arrange
    # Renew AdapterDataSource to ensure test independence
    cut = AdapterDataSource.__new__(AdapterDataSource)
    cut.new_data = True
    cut.new_data_lock = MagicMock()
    cut.double_buffer_read_index = pytest.gen.randint(0,1)
    pre_call_index = cut.double_buffer_read_index
    expected_result = MagicMock()
    cut.currentData = []
    if pre_call_index == 0:
        cut.currentData.append({'data': MagicMock()})
        cut.currentData.append({'data': expected_result})
    else:
        cut.currentData.append({'data': MagicMock()})
        cut.currentData.append({'data': expected_result})

    # Act
    result = cut.get_next()

    # Assert
    assert cut.new_data == False
    if pre_call_index == 0:
        assert cut.double_buffer_read_index == 1
    elif pre_call_index == 1:
        assert cut.double_buffer_read_index == 0
    else:
        assert False

    assert result == expected_result

def test_redis_adapter_AdapterDataSource_get_next_when_called_multiple_times_when_new_data_is_true():
    # Arrange
    # Renew AdapterDataSource to ensure test independence
    cut = AdapterDataSource.__new__(AdapterDataSource)
    cut.double_buffer_read_index = pytest.gen.randint(0,1)
    pre_call_index = cut.double_buffer_read_index

    # Act
    results = []
    num_calls = pytest.gen.randint(2,10) # arbitrary, 2 to 10
    for i in range(num_calls):
        cut.new_data = True
        results.append(cut.get_next())

    # Assert
    assert cut.new_data == False
    for i in range(num_calls):
        results[i] = cut.currentData[pre_call_index]['data']
        pre_call_index = (pre_call_index + 1) % 2
    assert cut.double_buffer_read_index == pre_call_index
    
def test_redis_adapter_AdapterDataSource_get_next_behavior_when_new_data_is_false_then_true(mocker):
    # Arrange
    # Renew AdapterDataSource to ensure test independence
    cut = AdapterDataSource.__new__(AdapterDataSource)
    cut.new_data_lock = MagicMock()
    cut.double_buffer_read_index = pytest.gen.randint(0,1)
    pre_call_index = cut.double_buffer_read_index
    expected_result = MagicMock()
    cut.currentData = []
    if pre_call_index == 0:
        cut.currentData.append({'data': MagicMock()})
        cut.currentData.append({'data': expected_result})
    else:
        cut.currentData.append({'data': MagicMock()})
        cut.currentData.append({'data': expected_result})

    num_falses = pytest.gen.randint(1, 10)
    side_effect_list = [False] * num_falses
    side_effect_list.append(True)

    print(side_effect_list)
    cut.new_data = PropertyMock()
    cut.new_data.side_effect=side_effect_list
    mocker.patch('onair.src.run_scripts.redis_adapter.time.sleep')

    # Act
    result = cut.get_next()

    # Assert
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
def test_redis_adapter_AdapterDataSource_has_more_returns_True():
    cut = AdapterDataSource.__new__(AdapterDataSource)
    assert cut.has_more
