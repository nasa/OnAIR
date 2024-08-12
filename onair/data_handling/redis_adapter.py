# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the
# Administrator of the National Aeronautics and Space Administration.
# No copyright is claimed in the United States under Title 17, U.S. Code.
# All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

"""
redis_adapter AdapterDataSource class

Receives messages from REDIS server, serves as a data source for sim.py
"""

import threading
import time
import redis
import json

from onair.data_handling.on_air_data_source import OnAirDataSource
from onair.data_handling.on_air_data_source import ConfigKeyError
from onair.data_handling.tlm_json_parser import parseJson
from onair.src.util.print_io import *
from onair.data_handling.parser_util import *


class DataSource(OnAirDataSource):

    def __init__(self, data_file, meta_file, ss_breakdown=False):
        super().__init__(data_file, meta_file, ss_breakdown)
        self.new_data_lock = threading.Lock()
        self.new_data = False
        self.servers = []
        self.currentData = []
        self.currentData.append(
            {"headers": self.order, "data": list("-" * len(self.order))}
        )
        self.currentData.append(
            {"headers": self.order, "data": list("-" * len(self.order))}
        )
        self.double_buffer_read_index = 0
        self.connect()

    def connect(self):
        """Establish connection to REDIS server."""
        print_msg('Redis adapter connecting to server...')
        for idx, server_config in enumerate(self.server_configs):
            server_config_keys = server_config.keys()
            if 'address' in server_config_keys:
                address = server_config['address']
            else:
                address = 'localhost'

            if 'port' in server_config_keys:
                port = server_config['port']
            else:
                port = 6379

            if 'db' in server_config_keys:
                db = server_config['db']
            else:
                db = 0

            if 'password' in server_config_keys:
                password = server_config['password']
            else:
                password = ''

            #if there are subscriptions in this Redis server configuration's subscription key
            if len(server_config['subscriptions']) != 0:
                #Create the servers and append them to self.servers list
                self.servers.append(redis.Redis(address, port, db, password))

                try:
                    #Ping server to make sure we can connect
                    self.servers[-1].ping()
                    print_msg(f'... connected to server # {idx}!')

                    #Set up Redis pubsub function for the current server
                    pubsub = self.servers[-1].pubsub()

                    for s in server_config['subscriptions']:
                        pubsub.subscribe(s)
                        print_msg(f"Subscribing to channel: {s} on server # {idx}")
                    listen_thread = threading.Thread(target=self.message_listener, args=(pubsub,))
                    listen_thread.start()

                #This except will be hit if self.servers[-1].ping() threw an exception (could not properly ping server)
                except:
                    print_msg(f'Did not connect to server # {idx}. Not setting up subscriptions.', ['FAIL'])

            else:
                print_msg("No subscriptions given! Redis server not created")

    def parse_meta_data_file(self, meta_data_file, ss_breakdown):
        self.server_configs = []
        configs = extract_meta_data_handle_ss_breakdown(
            meta_data_file, ss_breakdown)
        meta = parseJson(meta_data_file)
        keys = meta.keys()

        # Setup redis server configuration
        #Checking if 'redis' exists
        if 'redis' in keys:
            count_server_config = 0
            #Checking if dictionaries within 'redis' key each have a 'subscription' key. Error will be thrown if not.
            for server_config in meta['redis']:
                redis_config_keys = server_config.keys()
                if ('subscriptions' in redis_config_keys) == False:
                    raise ConfigKeyError(f'Config file: \'{meta_data_file}\' ' \
                        f'missing required key \'subscriptions\' from {count_server_config} in key \'redis\'')
                count_server_config +=1

            #Saving all of Redis dictionaries from JSON file to self.server_configs
            self.server_configs = meta['redis']

        if 'order' in keys:
            self.order = meta['order']
        else:
            raise ConfigKeyError(f'Config file: \'{meta_data_file}\' ' \
                                'missing required key \'order\'')

        return configs

    def process_data_file(self, data_file):
        print("Redis Adapter ignoring file")

    def get_vehicle_metadata(self):
        return self.all_headers, self.binning_configs["test_assignments"]

    def get_next(self):
        """Provides the latest data from REDIS channel"""
        data_available = False

        while not data_available:
            with self.new_data_lock:
                data_available = self.has_data()

            if not data_available:
                time.sleep(0.01)

        read_index = 0
        with self.new_data_lock:
            self.new_data = False
            self.double_buffer_read_index = (self.double_buffer_read_index + 1) % 2
            read_index = self.double_buffer_read_index

        return self.currentData[read_index]["data"]

    def has_more(self):
        """Live connection should always return True"""
        return True

    def message_listener(self, pubsub):
        """Loop for listening for messages on channels"""
        for message in pubsub.listen():
            if message['type'] == 'message':
                channel_name = f"{message['channel'].decode()}"
                # Attempt to load message as json
                try:
                    data = json.loads(message["data"])
                except ValueError:
                    # Warn of non-json conforming channel data received
                    non_json_msg = (
                        f"Subscribed channel `{channel_name}' "
                        "message received but is not in json "
                        f'format.\nMessage:\n{message["data"]}'
                    )
                    print_msg(non_json_msg, ["WARNING"])
                    continue
                # Select the current data
                currentData = self.currentData[(self.double_buffer_read_index + 1) % 2]
                # turn all data points to unknown
                currentData["data"] = ["-" for _ in currentData["data"]]
                # Find expected keys for received channel
                expected_message_keys = [
                    k for k in currentData["headers"] if channel_name in k
                ]
                # Time is an expected key for all channels
                expected_message_keys.append("time")
                # Parse through the message keys for data points
                for key in list(data.keys()):
                    if key.lower() == "time":
                        header_string = key.lower()
                    else:
                        header_string = f"{channel_name}.{key}"
                    # Look for channel specific values
                    try:
                        index = currentData["headers"].index(header_string)
                        currentData["data"][index] = data[key]
                        expected_message_keys.remove(header_string)
                    # Unexpected key in data
                    except ValueError:
                        # warn user about key in data that is not in header
                        print_msg(
                            f"Unused key `{key}' in message "
                            f"from channel `{channel_name}.'",
                            ["WARNING"],
                        )
                with self.new_data_lock:
                    self.new_data = True
                # Warn user about expected keys missing from received data
                for k in expected_message_keys:
                    print_msg(
                        f"Message from channel `{channel_name}' "
                        f"did not contain `{k}' key\nMessage:\n"
                        f"{data}",
                        ["WARNING"],
                    )
            else:
                # Warn user about non message receipts
                print_msg(
                    f"Redis adapter: channel "
                    f"'{message['channel'].decode()}' received "
                    f"message type: {message['type']}.",
                    ["WARNING"],
                )
        # When listener loop exits warn user
        print_msg("Redis subscription listener exited.", ["WARNING"])

    def has_data(self):
        return self.new_data
