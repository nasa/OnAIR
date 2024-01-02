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

from onair.data_handling.on_air_data_source import OnAirDataSource
from ctypes import *
import sbn_python_client as sbn
import message_headers as msg_hdr

from onair.data_handling.parser_util import *

# Note: The double buffer does not clear between switching. If fresh data doesn't come in, stale data is returned (delayed by 1 frame)

# msgID_lookup_table format { msgID : [ "<APP NAME>" , msg_hdr.<data struct in message_headers.py> , "<data category>" ] }
msgID_lookup_table = {0x0894 : ["SAMPLE", msg_hdr.CSV_APP_CSVData_t]}

#,
#                      0x0887 : ["SAMPLE", msg_hdr.sample_data_power_t],
#                      0x0889 : ["SAMPLE", msg_hdr.sample_data_thermal_t],
#                      0x088A : ["SAMPLE", msg_hdr.sample_data_gps_t]}

def message_listener_thread():
    """Thread to listen for incoming messages from SBN"""

    while(True):
        generic_recv_msg_p = POINTER(sbn.sbn_data_generic_t)()
        sbn.recv_msg(generic_recv_msg_p)

        msgID = generic_recv_msg_p.contents.TlmHeader.Primary.StreamId
        app_name, data_struct = msgID_lookup_table[msgID]

        recv_msg_p = POINTER(data_struct)()
        recv_msg_p.contents = generic_recv_msg_p.contents
        recv_msg = recv_msg_p.contents 
       
        # prints out the data from the message to the terminal
        print(", ".join([field_name + ": " + str(getattr(recv_msg, field_name)) for field_name, field_type in recv_msg._fields_[1:]]))

        # TODO: Lock needed here?
        get_current_data(recv_msg, data_struct, app_name)

def get_current_data(recv_msg, data_struct, app_name):
    # TODO: Lock needed here?
    current_buffer = DataSource.currentData[(DataSource.double_buffer_read_index + 1) %2]
    secondary_header = recv_msg.TlmHeader.Secondary

    #gets seconds from header and adds to current buffer
    start_time = datetime.datetime(1969, 12, 31, 20) # January 1, 1980
    seconds = secondary_header.Seconds
    subseconds = secondary_header.Subseconds
    curr_time = seconds + (2**(-32) * subseconds) # a subsecond is equal to 2^-32 second
    time = start_time + datetime.timedelta(seconds=curr_time)
    str_time = time.strftime("%Y-%j-%H:%M:%S.%f")
    current_buffer['data'][0] = str_time

    for field_name, field_type in data_struct._fields_[1:]:
        header_name = app_name + "." + data_struct.__name__ + "." + str(field_name)
        idx = current_buffer['headers'].index(header_name)
        data = str(getattr(recv_msg, field_name))
        current_buffer['data'][idx] = data

    with DataSource.new_data_lock:
        DataSource.new_data = True


class DataSource(OnAirDataSource):
    # Data structure
    # TODO: Make init data structure better
    # TODO: This should be in an __init__ function
    currentData = []

    for x in range(0,2):
        currentData.append({'headers' : [], 'data' : []})
        #print("Index {}".format(x))

        # First element is always the time, set to a dummy value here
        currentData[x]['headers'].append('TIME')
        currentData[x]['data'].append('2000-001-12:00:00.000000000')

        for msgID in msgID_lookup_table.keys():
            app_name, data_struct = msgID_lookup_table[msgID]
            struct_name = data_struct.__name__
            for field_name, field_type in data_struct._fields_[1:]:
                currentData[x]['headers'].append(app_name + "." + struct_name + "." + str(field_name))
            currentData[x]['data'].extend([0]*len(data_struct._fields_[1:])) #initialize all the data arrays with zero

    new_data_lock = threading.Lock()
    new_data = False
    double_buffer_read_index = 0

    def connect(self):
        """Establish connection to SBN and launch listener thread."""
        time.sleep(2)
        os.chdir("cf")
        sbn.sbn_load_and_init()
        print("SBN_Adapter Running")

        # Launch thread to listen for messages
        self.listener_thread = threading.Thread(target=message_listener_thread)
        self.listener_thread.start()

        # subscribe to message IDs
        for msgID in msgID_lookup_table.keys():
            sbn.subscribe(msgID)

    def subscribe_message(self, msgid):
        """Specify cFS message id to listen for. Unstable"""
        
        if isinstance(msgid, list):
            for mID in msgid:
                sbn.subscribe(mID)
        else:
            sbn.subscribe(msgid)

    def parse_meta_data_file(self, meta_data_file, ss_breakdown):
        # TODO: may want to parse sbn specific meta data here, like message ids
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
            with DataSource.new_data_lock:
                data_available = DataSource.new_data

            if not data_available:
                time.sleep(0.01)

        read_index = 0
        with DataSource.new_data_lock:
            DataSource.new_data = False
            DataSource.double_buffer_read_index = (DataSource.double_buffer_read_index + 1) % 2
            read_index = DataSource.double_buffer_read_index

        print("Reading buffer: {}".format(read_index))
        return self.currentData[read_index]['data']

    def has_more(self):
        """Returns true if the adapter has more data.
           For now always true: connection should be live as long as cFS is running.
           TODO: allow to detect if cFS/the connection has died"""
        return True
