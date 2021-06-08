"""
CSV Parser
"""

import csv
import time
import os
import re
import time

from src.util.file_io import *
from src.util.print_io import *
from src.data_handling.parsers.parser_util import * 

import pandas as pd

class CSV:
    def __init__(self, rawDataFilepath = '', 
                      metadataFilepath = '', 
                             dataFiles = '', 
                           configFiles = '', 
                          ss_breakdown = False):
        """An initial parsing needs to happen in order to use the parser classes
           This means that, if you want to use this class to parse in real time, 
           it needs to at least have seen one sample of the anticipated format """
        
        self.raw_data_file_path = rawDataFilepath
        self.metadata_file_path = metadataFilepath
        self.all_headers = ''
        self.sim_data = ''
        self.binning_configs = ''

        if (dataFiles != '') and (configFiles != ''):
            labels, data = self.parse_csv_data(str2lst(dataFiles)[0])
            self.all_headers = labels
            self.sim_data = data

            configs = self.parse_config_data(str2lst(configFiles)[0], ss_breakdown)

            self.binning_configs = {}
            self.binning_configs['subsystem_assignments'] = {}
            self.binning_configs['test_assignments'] = {}
            self.binning_configs['description_assignments'] = {}

            for data_file in ast.literal_eval(dataFiles):
                self.binning_configs['subsystem_assignments'][data_file] = configs['subsystem_assignments'][data_file]
                self.binning_configs['test_assignments'][data_file] = configs['test_assignments'][data_file]
                self.binning_configs['description_assignments'][data_file] = configs['description_assignments'][data_file]

##### INITIAL PROCESSING ####
    def parse_csv_data(self, dataFile):
        
        dataset = pd.read_csv(os.path.join(self.raw_data_filepath, dataFile), delimiter=',', header=0)
        dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]
        all_headers = list(dataset.columns.values)
        all_data = {}
        for index, row in dataset.iterrows():
            all_data[index] = {}
            all_data[dataFile] = list(row)
        return all_headers, all_data 

    def parse_config_data(self, configFile, ss_breakdown):
        parsed_configs = extract_configs(self.metadata_file_path, [configFile])
        if ss_breakdown == False:
            num_elements = len(parsed_configs['subsystem_assignments'][process_filepath(configFile)])
            parsed_configs['subsystem_assignments'][process_filepath(configFile)] = [['MISSION'] for elem in range(num_elements)]
        return parsed_configs
##### GETTERS ##################################

    def get_sim_data(self):
        return self.all_headers, self.sim_data, self.binning_configs
    