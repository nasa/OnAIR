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

from .on_air_parser import OnAirParser
from ...src.util.print_io import *
from .parser_util import * 

class CSV(OnAirParser):
    def pre_process_data(self, dataFiles):
        pass

    # TODO: This should go away, only one data file
    def process_data_per_data_file(self, data_file):
        labels, data = self.parse_csv_data(data_file)
        self.all_headers = labels
        self.sim_data = data

##### INITIAL PROCESSING ####
    def parse_csv_data(self, dataFile):
        #Read in the data set
        dataset = pd.read_csv(os.path.join(self.raw_data_filepath, dataFile), delimiter=',', header=0, dtype=str)
        dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]

        all_headers = list(dataset.columns.values)
        #Find the 'Time' header in the list in order to match 42 file formatting 
        # Converting
        upperCaseStringHeaders = [x.upper().strip() for x in all_headers if isinstance(x, str)]
        #Search for TIME header in list of uppercase string headers, if it's not there it should return a valueerror, set index to -1
        try:
            timeIndex = upperCaseStringHeaders.index('TIME')
        except ValueError:
            timeIndex = -1
        #Initialize the entire data dictionary
        all_data = []
        for index, row in dataset.iterrows():
            rowVals = floatify_input(list(row))
            all_data.append(floatify_input(list(row)))

        return all_headers, all_data 

    def parse_config_data(self, configFile, ss_breakdown):
        parsed_configs = extract_configs(self.metadata_filepath, configFile)
        if ss_breakdown == False:
            num_elements = len(parsed_configs['subsystem_assignments'])
            parsed_configs['subsystem_assignments'] = [['MISSION'] for elem in range(num_elements)]
        return parsed_configs

##### GETTERS ##################################

    def get_sim_data(self):
        return self.all_headers, self.sim_data, self.binning_configs

    def get_just_data(self):
        return self.sim_data

    def get_vehicle_metadata(self):
        print(self.binning_configs['test_assignments'])
        return self.all_headers, self.binning_configs['test_assignments']
