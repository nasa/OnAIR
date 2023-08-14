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

from ...data_handling.data_source import DataSource

class AdapterDataSource(DataSource):

    def __init__(self, data=[]):
        super().__init__(data)
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

    def connect(self):
        """Establish connection to REDIS server."""
        self.server = redis.Redis(self.address, self.port, self.db)


    def subscribe_message(self, channel):
        """Subscribe to REDIS message channel and launch listener thread."""
        if self.server.ping():
            self.pubsub = self.server.pubsub()
            self.pubsub.subscribe(channel)

            listen_thread = threading.Thread(target=self.message_listener)
            listen_thread.start()


    def get_next(self):
        """Provides the latest data from REDIS channel"""
        data_available = False

        while not data_available:
            with self.new_data_lock:
                data_available = self.new_data

            if not data_available:
                time.sleep(0.01)

        read_index = 0
        with self.new_data_lock:
            self.new_data = False
            self.double_buffer_read_index = (self.double_buffer_read_index + 1) % 2
            read_index = self.double_buffer_read_index

        print("Reading buffer: {}".format(read_index))
        return self.currentData[read_index]['data']

    def has_more(self):
        """Live connection should always return True"""
        return True

    def message_listener(self):
        """Loop for listening for messages on channel"""
        for message in self.pubsub.listen():
            print("Received from REDIS: ", message)
            if message['type'] == 'message':
                data = json.loads(message['data'])

                currentData = self.currentData[(self.double_buffer_read_index + 1) %2]
                currentData['headers'] = list(data.keys())
                currentData['data'] = list(data.values())

                with self.new_data_lock:
                    self.new_data = True

