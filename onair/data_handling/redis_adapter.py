# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the
# Administrator of the National Aeronautics and Space Administration.
# No copyright is claimed in the United States under Title 17, U.S. Code.
# All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

"""redis_adapter module

This module contains a DataSource class, which serves as a data source
for sim.py by receiving messages from one or more REDIS servers. It
implements the OnAirDataSource interface and provides functionality for
connecting to REDIS servers, subscribing to channels, and processing
incoming data.

The module utilizes Redis, threading, and JSON libraries to handle
server connections and data parsing.
"""

import threading
import time
import json
import redis

from onair.data_handling.on_air_data_source import OnAirDataSource
from onair.data_handling.on_air_data_source import ConfigKeyError
from onair.data_handling.tlm_json_parser import parseJson
from onair.src.util.print_io import print_msg
from onair.data_handling.parser_util import extract_meta_data_handle_ss_breakdown


class DataSource(OnAirDataSource):
    """Implements OnAirDataSource interface for receiving data from REDIS servers.

    This class provides the following functionality:
    - Establishes connections to one or more REDIS servers
    - Subscribes to specified channels on each server
    - Listens for incoming messages on subscribed channels
    - Parses and processes received data
    - Provides methods to access the latest data received

    The class uses double-buffering for thread-safe access to recent data.
    """

    def __init__(self, data_file, meta_file, ss_breakdown=False):
        """Initialize the DataSource object.

        Parameters
        ----------
        data_file : str
            Path to the data file (not used in Redis adapter).
        meta_file : str
            Path to the metadata file containing Redis server configurations.
        ss_breakdown : bool, optional
            Flag to indicate whether to handle subsystem breakdown, by default False.
            Flag to indicate whether to handle subsystem breakdown, by default
            False.

        Notes
        -----
        This method performs the following tasks:
        1. Initializes the parent class.
        2. Sets up threading lock and new data flag.
        3. Initializes lists for servers and current data.
        4. Creates a double buffer for current data storage.
        5. Connects to Redis servers specified in the metadata file.
        """
        super().__init__(data_file, meta_file, ss_breakdown)
        self.new_data_lock = threading.Lock()
        self.new_data = False
        self.servers = []
        self.current_data = []
        self.current_data.append(
            {"headers": self.order, "data": list("-" * len(self.order))}
        )
        self.current_data.append(
            {"headers": self.order, "data": list("-" * len(self.order))}
        )
        self.double_buffer_read_index = 0
        self.connect()

    def connect(self):
        """Connect to Redis servers and set up subscriptions.

        This method iterates through the server configurations, attempts to
        connect to each Redis server, and sets up subscriptions for the
        specified channels.

        For each server configuration:
        1. Extracts connection details (address, port, db, password).
        2. Creates a Redis connection if subscriptions are specified.
        3. Pings the server to ensure connectivity.
        4. Sets up a pubsub object and subscribes to specified channels.
        5. Starts a listener thread for each server connection.

        When a connection fails, an error message is output, and the method
        continues to the next server configuration.
        """
        print_msg("Redis adapter connecting to server...")
        for idx, server_config in enumerate(self.server_configs):
            address = server_config.get("address", "localhost")
            port = server_config.get("port", 6379)
            db = server_config.get("db", 0)
            password = server_config.get("password", "")

            # if there are subscriptions in this Redis server configuration's subscription key
            if len(server_config["subscriptions"]) != 0:
                # Create the servers and append them to self.servers list
                self.servers.append(redis.Redis(address, port, db, password))

                try:
                    # Ping server to make sure we can connect
                    self.servers[-1].ping()
                    print_msg(f"... connected to server # {idx}!")

                    # Set up Redis pubsub function for the current server
                    pubsub = self.servers[-1].pubsub()

                    for s in server_config["subscriptions"]:
                        pubsub.subscribe(s)
                        print_msg(f"Subscribing to channel: {s} on server # {idx}")
                    listen_thread = threading.Thread(
                        target=self.message_listener, args=(pubsub,)
                    )
                    listen_thread.start()

                # This except will be hit if self.servers[-1].ping()
                # threw an exception (could not properly ping server)
                except (
                    redis.exceptions.ConnectionError,
                    redis.exceptions.TimeoutError,
                ) as e:
                    error_type = type(e).__name__
                    error_message = str(e)
                    print_msg(
                        f"Did not connect to server # {idx} due to {error_type}: {error_message}"
                        f"\nNot setting up subscriptions.",
                        ["FAIL"],
                    )

            else:
                print_msg("No subscriptions given! Redis server not created")

    def parse_meta_data_file(self, meta_data_file, ss_breakdown):
        """
        Parse the metadata file and extract configuration information.

        Parameters
        ----------
        meta_data_file : str
            Path to the metadata file.
        ss_breakdown : bool
            Flag to indicate whether to handle subsystem breakdown.

        Returns
        -------
        dict
            Extracted configuration information.

        Raises
        ------
        ConfigKeyError
            If required keys are missing in the metadata file.

        Notes
        -----
        This method performs the following tasks:
        1. Extracts metadata and handles subsystem breakdown.
        2. Parses the JSON content of the metadata file.
        3. Validates and extracts Redis server configurations.
        4. Extracts the 'order' key from the metadata.

        The extracted Redis server configurations are stored in
        `self.server_configs`.
        The 'order' key is stored in `self.order`.
        """
        self.server_configs = []
        configs = extract_meta_data_handle_ss_breakdown(meta_data_file, ss_breakdown)
        meta = parseJson(meta_data_file)
        keys = meta.keys()

        # Setup redis server configuration
        # Checking if 'redis' exists
        if "redis" in keys:
            count_server_config = 0
            # Checking if dictionaries within 'redis' key each have a 'subscription' key.
            for server_config in meta["redis"]:
                redis_config_keys = server_config.keys()
                if not "subscriptions" in redis_config_keys:
                    raise ConfigKeyError(
                        f"Config file: '{meta_data_file}' "
                        f"missing required key 'subscriptions' from {count_server_config}"
                        + " in key 'redis'"
                    )
                count_server_config += 1

            # Saving all of Redis dictionaries from JSON file to self.server_configs
            self.server_configs = meta["redis"]

        if "order" in keys:
            self.order = meta["order"]
        else:
            raise ConfigKeyError(
                f"Config file: '{meta_data_file}' " "missing required key 'order'"
            )

        return configs

    def process_data_file(self, data_file):
        """Process the data file (not used in Redis Adapter).

        This method is not used in the Redis Adapter and simply prints a
        message indicating that the file is being ignored.

        Parameters
        ----------
        data_file : str
            Path to the data file (not used).

        Notes
        -----
        Data is received through Redis subscriptions rather than from a file,
        so this method does not perform any actual processing.
        """
        print("Redis Adapter ignoring file")

    def get_vehicle_metadata(self):
        """Get the vehicle metadata for headers and test assignments.

        Returns
        -------
        tuple
            A tuple containing two elements:
            - all_headers : list
                A list of all headers for the vehicle data.
            - test_assignments : dict
                A dictionary containing the test assignments from the binning
                configurations.

        Notes
        -----
        This method returns the metadata necessary for processing and organizing
        vehicle data. The all_headers list provides information about the
        structure of the data, while the test_assignments dictionary contains
        information about how tests are assigned or grouped.
        """
        return self.all_headers, self.binning_configs["test_assignments"]

    def get_next(self):
        """Retrieve the next available data from the buffer.

        This method waits for new data to become available, then returns it.
        It uses a double-buffering technique to ensure thread-safe access to
        the data.

        Returns
        -------
        list
            The latest data retrieved from the buffer.

        Notes
        -----
        This method blocks until new data is available. It uses a short sleep
        (10 milliseconds) between checks to reduce CPU usage while waiting.

        The double-buffering technique involves:
        1. Waiting for the `new_data` flag to become True.
        2. Resetting the `new_data` flag to False.
        3. Updating the read index for the double buffer.
        4. Returning the data from the current read buffer.

        This approach allows one buffer to be read while the other is being
        written to, ensuring data consistency and thread safety.
        """
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

        return self.current_data[read_index]["data"]

    def has_more(self):
        """Check if more data is available.

        This method always returns True for the Redis adapter, as it
        continuously listens for new messages.

        Returns
        -------
        bool
            Always True, indicating that more data can potentially be received.

        Notes
        -----
        This method is part of the OnAirDataSource interface. For the Redis
        adapter, it always returns True because the adapter continuously
        listens for new messages from the subscribed Redis channels.
        """
        return True

    def message_listener(self, pubsub):
        """Listen for messages from Redis pubsub and process them.

        This method continuously listens for messages from the Redis pubsub
        connection, processes JSON messages, updates the current data buffer,
        handling various error conditions.

        Parameters
        ----------
        pubsub : redis.client.PubSub
            A Redis pubsub object for subscribing to channels and receiving
            messages.

        Notes
        -----
        The method performs the following tasks:
        1. Listens for messages from the pubsub connection.
        2. Processes messages of type "message".
        3. Attempts to parse the message data as JSON.
        4. Updates the current data buffer with new values.
        5. Handles missing or unexpected keys in the message data.
        6. Sets a flag to indicate new data is available.
        7. Warns about non-message type receipts.

        The method continues to run until the pubsub connection is closed or
        an error occurs. If the listener loop exits, a warning is issued.

        Warnings
        --------
        Warnings are issued for the following conditions:
        - Non-JSON conforming message data
        - Unexpected keys in the message data
        - Expected keys missing from the message data
        - Non-message type receipts
        - Listener loop exit
        """
        for message in pubsub.listen():
            if message["type"] == "message":
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
                current_data = self.current_data[
                    (self.double_buffer_read_index + 1) % 2
                ]
                # turn all data points to unknown
                current_data["data"] = ["-" for _ in current_data["data"]]
                # Find expected keys for received channel
                expected_message_keys = [
                    k for k in current_data["headers"] if channel_name in k
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
                        index = current_data["headers"].index(header_string)
                        current_data["data"][index] = data[key]
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
        """
        Check if new data is available.

        Returns
        -------
        bool
            The value of the new_data flag, which should be True if new data is
            available, False otherwise.
        """
        return self.new_data
