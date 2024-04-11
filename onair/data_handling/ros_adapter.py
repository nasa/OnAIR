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
import json
from functools import partial

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String

from onair.data_handling.on_air_data_source import OnAirDataSource
from onair.data_handling.tlm_json_parser import parseJson
from onair.src.util.print_io import *
from onair.data_handling.parser_util import *

# ROS message types must be mapped here from string representations that can be referenced in the JSON config
ROS_MSG_TYPE_MAPPINGS = {
    "Bool": Bool,
    "String": String,
}


class ROSSubscriber(Node):
    def __init__(self, name, topic_definitions):
        super().__init__(node_name=name)
        self.sub_list = []
        for topic_definition in topic_definitions:
            message_type = ROS_MSG_TYPE_MAPPINGS[topic_definition['message_type']]

            self.sub_list.append(self.create_subscription(
                topic=topic_definition['topic'],
                msg_type=message_type,
                callback=partial(self.listener_callback, topic=topic_definition['topic']),
                qos_profile=10
            ))
    
    def listener_callback(self, msg, topic):
        self.received_topic = topic
        self.received_msg = msg

class DataSource(OnAirDataSource):

    def __init__(self, data_file, meta_file, ss_breakdown = False):
        super().__init__(data_file, meta_file, ss_breakdown)

        node_name = 'onair_subscriber'
        self.topic_active = False
        
        self.configs = self.parse_meta_data_file(self.meta_data_file, ss_breakdown)
        self.low_level_data = []

        self.low_level_data.append({'headers':self.order,
                                 'data':list('-' * len(self.order))})
        self.low_level_data.append({'headers':self.order,
                                 'data':list('-' * len(self.order))})

        self.double_buffer_read_index = 0
        
        for topic_def in self.topic_definitions:
            print(bcolors.OKBLUE + f'OnAIR subscribed to topic {topic_def["topic"]}' + bcolors.ENDC)

        rclpy.init()
        self.sub_node = ROSSubscriber(name=node_name, topic_definitions=self.topic_definitions)

    def parse_meta_data_file(self, meta_data_file, ss_breakdown):
        configs = extract_meta_data_handle_ss_breakdown(meta_data_file, ss_breakdown)
        meta = parseJson(meta_data_file)
        if 'ros_subscriptions' in meta.keys():
            self.topic_definitions = meta['ros_subscriptions']                
        else:
            self.topic_definitions = []
        
        self.order = meta['order']
        
        return configs

    def process_data_file(self, data_file):
        pass # ROS Adapter does not use data file

    def get_vehicle_metadata(self):
        return self.all_headers, self.binning_configs['test_assignments']

    def get_next(self):
        """Provides the latest data from REDIS channel"""

        self.double_buffer_read_index = (
                self.double_buffer_read_index + 1) % 2
        low_level_data = self.low_level_data[(self.double_buffer_read_index + 1) % 2]

        if not self.topic_active:
            print(bcolors.WARNING + f"Waiting on heartbeat from subscribed topics" + bcolors.ENDC)
            rclpy.spin_once(self.sub_node)
            self.topic_active = True
            header_string = self.sub_node.received_topic
            index = low_level_data['headers'].index(header_string)
            low_level_data['data'][index] = self.sub_node.received_msg.data
        else:
            rclpy.spin_once(self.sub_node)
            header_string = self.sub_node.received_topic
            index = low_level_data['headers'].index(header_string)
            low_level_data['data'][index] = self.sub_node.received_msg.data
        
        self.low_level_data[self.double_buffer_read_index] = low_level_data
        return low_level_data['data']

    def has_more(self):
        """Live connection should always return True"""
        return True
    