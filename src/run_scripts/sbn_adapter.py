"""
AdapterDataSource class

Receives messages from SBN, serves as a data source for sim.py
"""

import threading
import time
import datetime
import os

from data_handling.data_source import DataSource
from ctypes import *
import sbn_client as sbn
import message_headers as msg_hdr

# Note: The double buffer does not clear between switching. If fresh data doesn't come in, stale data is returned (delayed by 1 frame)

# msgID_lookup_table format { msgID : [ "<APP NAME>" , msg_hdr.<data struct in message_headers.py> , "<data category>" ] }
msgID_lookup_table = {0x0885 : ["SAMPLE", msg_hdr.sample_data_tlm_t],
                      0x0887 : ["SAMPLE", msg_hdr.sample_data_power_t],
                      0x0889 : ["SAMPLE", msg_hdr.sample_data_thermal_t],
                      0x088A : ["SAMPLE", msg_hdr.sample_data_gps_t]}

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
    current_buffer = AdapterDataSource.currentData[(AdapterDataSource.double_buffer_read_index + 1) %2]
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

    with AdapterDataSource.new_data_lock:
        AdapterDataSource.new_data = True


class AdapterDataSource(DataSource):
    # Data structure (shares code with binner.py)
    # TODO: Make init data structure better
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

    def get_next(self):
        """Provides the latest data from SBN in a dictionary of lists structure.
        Returned data is safe to use until the next get_next call.
        Blocks until new data is available."""

        data_available = False

        while not data_available:
            with AdapterDataSource.new_data_lock:
                data_available = AdapterDataSource.new_data

            if not data_available:
                time.sleep(0.01)

        read_index = 0
        with AdapterDataSource.new_data_lock:
            AdapterDataSource.new_data = False
            AdapterDataSource.double_buffer_read_index = (AdapterDataSource.double_buffer_read_index + 1) % 2
            read_index = AdapterDataSource.double_buffer_read_index

        print("Reading buffer: {}".format(read_index))
        return self.currentData[read_index]['data']

    def has_more(self):
        """Returns true if the adapter has more data. Always true: connection should be live as long as cFS is running"""
        return True
