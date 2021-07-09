"""
CSV Parser
"""

import csv
import time
import os
import re
import time

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
            # Setup headers, data
            headers_dict = {}
            data_dict = {}
            
            # Parse data across multiple files
            for data_file in str2lst(dataFiles):
                labels, data = self.parse_csv_data(data_file)
                # Header format : { Filename : ['Header', 'Another Header', 'Etc.']}
                headers_dict[data_file] = labels[data_file] #The key for labels is the file name, so we're able to add that to our "big" dictionary
                # Data format : { 'index': { Filename : ['Data_Point', 'Another Data_Point', 'Etc.']}}
                for key in data:
                    # If the key does not exist in the data dictionary already, then data_dict[key][data_file] will give an error
                    # In order to skirt this issue, if the key does not exist create an empty dictionary for it, then give it the data file
                    if key in data_dict:
                        data_dict[key][data_file] = data[key][data_file]
                    else:
                        data_dict[key] = {}
                        data_dict[key][data_file] = data[key][data_file]
                        
            self.all_headers = headers_dict
            self.sim_data = data_dict

            # Setup binning config information
            self.binning_configs = {}
            self.binning_configs['subsystem_assignments'] = {}
            self.binning_configs['test_assignments'] = {}
            self.binning_configs['description_assignments'] = {}

            for config_file in str2lst(configFiles):
                # Config format {'subsystem' : {data_file : ['Etc.']}}
                config = self.parse_config_data_CSV(config_file, ss_breakdown)
                # Although this is a for loop, it should only execute once because the config currently has only read in one single file 
                for data_file_key in config['subsystem_assignments']:
                    self.binning_configs['subsystem_assignments'][data_file_key] = config['subsystem_assignments'][data_file_key]
                    self.binning_configs['test_assignments'][data_file_key] = config['test_assignments'][data_file_key]
                    self.binning_configs['description_assignments'][data_file_key] = config['description_assignments'][data_file_key]

            

##### INITIAL PROCESSING ####
    def parse_csv_data(self, dataFile):
        #Read in the data set
        dataset = pd.read_csv(os.path.join(self.raw_data_file_path, dataFile), delimiter=',', header=0, dtype=str)
        dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]
        #Get headers from dataset and format them into a dictionary
        # {fileName: [Header1, Header2, ...]}
        headersDict = {}
        headersDict[dataFile] = list(dataset.columns.values)
        all_headers = headersDict
        #Find the 'Time' header in the list in order to match 42 file formatting 
        # Converting
        upperCaseStringHeaders = [x.upper().strip() for x in list(dataset.columns.values) if isinstance(x, str)]
        #Search for TIME header in list of uppercase string headers, if it's not there it should return a valueerror, set index to -1
        try:
            timeIndex = upperCaseStringHeaders.index('TIME')
        except ValueError:
            timeIndex = -1
        #Initialize the entire data dictionary
        all_data = {}
        for index, row in dataset.iterrows():
            rowVals = list(row)
            innerStructure = {dataFile : list(row)}
            #If a time header doesn't exist, just assume normal indexing
            if (timeIndex == -1):
                all_data[index] = innerStructure
            else:
                all_data[rowVals[timeIndex]] = innerStructure
        return all_headers, all_data 

    def parse_config_data_CSV(self, configFile, ss_breakdown):
        parsed_configs = extract_configs(self.metadata_file_path, [configFile], csv=True)
        if ss_breakdown == False:
            num_elements = len(parsed_configs['subsystem_assignments'][process_filepath(configFile, csv=True)])
            parsed_configs['subsystem_assignments'][process_filepath(configFile, csv=True)] = [['MISSION'] for elem in range(num_elements)]
        return parsed_configs

##### GETTERS ##################################

    def get_sim_data(self):
        return self.all_headers, self.sim_data, self.binning_configs
    