# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

"""
SBN_Adapter class

Receives messages from SBN, serves as a data source for sim.py
"""

import threading
import time
import datetime
import os
import json

from onair.data_handling.on_air_data_source import OnAirDataSource
from ctypes import *
import sbn_python_client as sbn
import message_headers as msg_hdr

from onair.data_handling.parser_util import *

# Note: The double buffer does not clear between switching. If fresh data doesn't come in, stale data is returned (delayed by 1 frame)

class DataSource(OnAirDataSource):

    def __init__(self, data_file, meta_file, ss_breakdown = False):
        super().__init__(data_file, meta_file, ss_breakdown);

        self.new_data_lock = threading.Lock()
        self.new_data = False
        self.double_buffer_read_index = 0
        self.connect()

    def connect(self):
        """Establish connection to SBN and launch listener thread."""
        time.sleep(2)
        os.chdir("cf")
        sbn.sbn_load_and_init()
        os.chdir("../")
        print("SBN_Adapter Running")

        # Launch thread to listen for messages
        self.listener_thread = threading.Thread(target=self.message_listener_thread)
        self.listener_thread.start()

        # subscribe to message IDs
        for msgID in self.msgID_lookup_table.keys():
            sbn.subscribe(msgID)

    def gather_field_names(self, field_name, field_type):
        field_names = []
        if "message_headers" in str(field_type):
            for sub_field_name, sub_field_type in field_type._fields_:
                field_names.append(self.gather_field_names(field_name + "." + sub_field_name, sub_field_type))
        else:
            #field_names.append(field_name)
            return field_name
        return field_names

    def parse_meta_data_file(self, meta_data_file, ss_breakdown):
        self.msgID_lookup_table = {}
        self.currentData = []

        # pull out message ids
        file = open(meta_data_file, 'rb')
        file_str = file.read()

        meta_config = json.loads(file_str)
        file.close()

        # Copy message ID table from .json, convert string hex to ints for ID
        for key in meta_config['channels']:
            self.msgID_lookup_table[int(key, 16)] = meta_config['channels'][key]

        # Use eval() to convert class name from .json to match with message_headers.py
        for key in self.msgID_lookup_table:
            msg_struct_name = self.msgID_lookup_table[key][1]
            self.msgID_lookup_table[key][1] = eval("msg_hdr." + msg_struct_name)

        # populate headers and reserve space for data
        for x in range(0,2):
            self.currentData.append({'headers':[], 'data':[]})

            for msgID in self.msgID_lookup_table.keys():
                app_name, data_struct = self.msgID_lookup_table[msgID]
                struct_name = data_struct.__name__
                # Skip the header, walk through the stuct
                for field_name, field_type in data_struct._fields_[1:]:
                    field_names = self.gather_field_names(app_name + "." + field_name, field_type)

                    for field_name in field_names:
                        self.currentData[x]['headers'].append(field_name)
                        self.currentData[x]['data'].append([0]) #initialize all the data arrays with zero

        return extract_meta_data_handle_ss_breakdown(meta_data_file, ss_breakdown)

    def process_data_file(self, data_file):
        print("SBN Adapter ignoring data file (telemetry should be live)")

    def get_vehicle_metadata(self):
        return self.all_headers, self.binning_configs['test_assignments']

    def get_next(self):
        """Provides the latest data from SBN in a dictionary of lists structure.
        Returned data is safe to use until the next get_next call.
        Blocks until new data is available."""

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

        return self.currentData[read_index]['data']

    def has_more(self):
        """Returns true if the adapter has more data.
           For now always true: connection should be live as long as cFS is running.
           TODO: allow to detect if cFS/the connection has died"""
        return True

    def message_listener_thread(self):
        """Thread to listen for incoming messages from SBN"""

        while(True):
            generic_recv_msg_p = POINTER(sbn.sbn_data_generic_t)()
            sbn.recv_msg(generic_recv_msg_p)

            msgID = generic_recv_msg_p.contents.TlmHeader.Primary.StreamId
            app_name, data_struct = self.msgID_lookup_table[msgID]

            recv_msg_p = POINTER(data_struct)()
            recv_msg_p.contents = generic_recv_msg_p.contents
            recv_msg = recv_msg_p.contents

            # prints out the data from the message to the terminal
            print(", ".join([field_name + ": " + str(getattr(recv_msg, field_name)) for field_name, field_type in recv_msg._fields_[1:]]))

            # TODO: Lock needed here?
            self.get_current_data(recv_msg, data_struct, app_name)

    def get_current_data(self, recv_msg, data_struct, app_name):
        # TODO: Lock needed here?
        current_buffer = self.currentData[(self.double_buffer_read_index + 1) %2]
        secondary_header = recv_msg.TlmHeader.Secondary

        #gets seconds from header and adds to current buffer
        start_time = datetime.datetime(1969, 12, 31, 20) # January 1, 1980
        seconds = secondary_header.Seconds
        subseconds = secondary_header.Subseconds
        curr_time = seconds + (2**(-32) * subseconds) # a subsecond is equal to 2^-32 second
        time = start_time + datetime.timedelta(seconds=curr_time)
        str_time = time.strftime("%Y-%j-%H:%M:%S.%f")
        current_buffer['data'][0] = str_time

        # Skip the header, walk through the stuct
        for field_name, field_type in recv_msg._fields_[1:]:
            field_names = self.gather_field_names(field_name, field_type)

            for name in field_names:
                idx = current_buffer['headers'].index(app_name + "." + name)
                # Pull the data out of the message buy walking down the nested types
                data = ""
                current_object = recv_msg
                for sub_type in name.split('.'):
                    print ("sub_type: " + sub_type)
                    current_object = getattr(current_object, sub_type)
                    data = str(current_object)
                    print("\tdata: " + data)
                current_buffer['data'][idx] = data

        with self.new_data_lock:
            self.new_data = True
