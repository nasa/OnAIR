# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
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
from onair.data_handling.tlm_json_parser import parseJson
from onair.src.util.print_io import *
from onair.data_handling.parser_util import *

class DataSource(OnAirDataSource):

    def __init__(self, data_file: str, meta_file: str, ss_breakdown: bool = False) -> None:
        super().__init__(data_file, meta_file, ss_breakdown)
        self.address = 'localhost'
        self.port = 6379
        self.db = 0
        self.server = None  
        self.new_data_lock = threading.Lock()
        self.new_data = False
        self.currentData = []
        self.currentData.append({'headers':None, 'data':None})
        self.currentData.append({'headers':None, 'data':None})
        self.double_buffer_read_index = 0
        self.connect()
        self.subscribe(self.subscriptions)

    def connect(self) -> None:
        """Establish connection to REDIS server."""
        print_msg('Redis adapter connecting to server...')
        self.server = redis.Redis(self.address, self.port, self.db)

        if self.server.ping():
            print_msg('... connected!')

    def subscribe(self, subscriptions: list) -> None:
        """Subscribe to REDIS message channel(s) and launch listener thread."""
        if len(subscriptions) != 0 and self.server.ping():
            self.pubsub = self.server.pubsub()

            for s in subscriptions:
                self.pubsub.subscribe(s)
                print_msg(f"Subscribing to channel: {s}")

            listen_thread = threading.Thread(target=self.message_listener)
            listen_thread.start()
        else:
            print_msg(f"No subscriptions given!")

    def parse_meta_data_file(self, meta_data_file: str, ss_breakdown: bool) -> dict:
        configs = extract_meta_data_handle_ss_breakdown(meta_data_file, ss_breakdown)
        meta = parseJson(meta_data_file)
        if 'redis_subscriptions' in meta.keys():
            self.subscriptions = meta['redis_subscriptions']
        else:
            self.subscriptions = []

        return configs

    def process_data_file(self, data_file: str) -> None:
        print("Redis Adapter ignoring file")

    def get_vehicle_metadata(self) -> tuple:
        return self.all_headers, self.binning_configs['test_assignments']

    def get_next(self) -> list:
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

        return self.currentData[read_index]['data']

    def has_more(self) -> bool:
        """Live connection should always return True"""
        return True

    def message_listener(self) -> None:
        """Loop for listening for messages on channel"""
        for message in self.pubsub.listen():
            if message['type'] == 'message':
                data = json.loads(message['data'])

                currentData = self.currentData[(self.double_buffer_read_index + 1) %2]
                currentData['headers'] = list(data.keys())
                currentData['data'] = list(data.values())

                with self.new_data_lock:
                    self.new_data = True

    def has_data(self) -> bool:
        return self.new_data
    