# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

"""
CSV Parser
"""

import os
import pandas as pd

from onair.data_handling.on_air_data_source import OnAirDataSource
from onair.src.util.print_io import *
from onair.data_handling.parser_util import *

class DataSource(OnAirDataSource):
    """
    A data source for processing CSV data.

    Args:
    --------
        data_file (str): The path to the raw data CSV file.
        meta_file (str): The path to the metadata file.
        ss_breakdown (bool): Whether to perform subsystem breakdown (default: False).
    """

    def process_data_file(self, data_file):
        """
        Process the data file and store it in the `sim_data` attribute.

        Args:
        --------
            data_file (str): The path to the raw data CSV file.
        """
        self.sim_data = self.parse_csv_data(data_file)
        self.frame_index = 0

##### INITIAL PROCESSING ####
    def parse_csv_data(self, data_file):
        """
        Parse the CSV data file and convert it into a list of lists.

        Args:
        --------
            data_file (str): The path to the CSV data file.

        Returns:
        --------
            list: A list of lists containing the parsed data.
        """
        #Read in the data set
        dataset = pd.read_csv(data_file, delimiter=',', header=0, dtype=str)
        dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]

        #Initialize the entire data dictionary
        all_data = []
        for index, row in dataset.iterrows():
            rowVals = floatify_input(list(row))
            all_data.append(floatify_input(list(row)))

        return all_data

    def parse_meta_data_file(self, meta_data_file, ss_breakdown):
        """
        Parse the metadata file and return binning configurations.

        Args:
        --------
            meta_data_file (str): The path to the metadata file.
            ss_breakdown (bool): Whether to perform subsystem breakdown.

        Returns:
        --------
            dict: A dictionary containing binning configurations.
        """
        parsed_meta_data = extract_meta_data(meta_data_file)
        if ss_breakdown == False:
            num_elements = len(parsed_meta_data['subsystem_assignments'])
            parsed_meta_data['subsystem_assignments'] = [['MISSION'] for elem in range(num_elements)]
        return parsed_meta_data

##### GETTERS ##################################
    def get_sim_data(self):
        """
        Get simulation data along with headers and binning configurations.

        Returns:
        --------
            tuple: A tuple containing headers, simulation data, and binning configurations.
        """
        return self.all_headers, self.sim_data, self.binning_configs

    def get_just_data(self):
        """
        Get only the simulation data.

        Returns:
        --------
            list: A list containing the simulation data.
        """
        return self.sim_data

    def get_vehicle_metadata(self):
        """
        Get vehicle metadata, including headers and test assignments.

        Returns:
        --------
            tuple: A tuple containing headers and test assignments.
        """
        return self.all_headers, self.binning_configs['test_assignments']

    # Get the data at self.index and increment the index
    def get_next(self):
        """
        Get the data at the current index and increment the index.

        Returns:
        -------
            list: A list containing the data at the current index.
        """
        self.frame_index = self.frame_index + 1
        return self.sim_data[self.frame_index - 1]

    # Return whether or not the index has finished traveling through the data
    def has_more(self):
        """
        Check if there are more frames to process.

        Returns:
        --------
            bool: True if there are more frames, False otherwise.
        """
        return self.frame_index < len(self.sim_data)
