"""
42 Parser

Only headers are parsed on init. This parser is used for the cFS adapter, which does not
have real data upon start up.
"""

import csv
import time
import os
import re
import time

from src.util.print_io import *
from src.data_handling.parsers.parser_util import * 

class FortyTwo:
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
            # Setup binning config information
            self.binning_configs = {}
            self.binning_configs['subsystem_assignments'] = {}
            self.binning_configs['test_assignments'] = {}
            self.binning_configs['description_assignments'] = {}

            config = self.parse_config_data(str2lst(configFiles)[0], ss_breakdown)
            # Setup headers, data
            headers_dict = {}
            data_dict = {}

            # Parse data across multiple files
            for data_file in str2lst(dataFiles):
                labels, data = self.parse_sim_data(data_file)
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
                self.binning_configs['subsystem_assignments'][data_file] = config['subsystem_assignments']
                self.binning_configs['test_assignments'][data_file]= config['test_assignments']
                self.binning_configs['description_assignments'][data_file] = config['description_assignments']
                        
            self.all_headers = headers_dict
            self.sim_data = data_dict

    ##### INITIAL PROCESSING ####
    def parse_sim_data(self, dataFile):
        headers = []
        frames = []

        txt_file = open(self.raw_data_file_path + dataFile,"r+")

        data_str = txt_file.read()
        txt_file.close()

        dataPts = data_str.split('[EOF]\n\n') # Get each frame in a string
        dataPts = [elem.strip() for elem in dataPts]
        dataPts.remove('')
        headers = self.parse_headers(dataPts[0])

        # Process into binning format
        all_headers = {}

        all_headers[dataFile] = headers

        return all_headers, {}

    def parse_headers(self, frame):
        headers = []
        data = frame.replace('SC[0].', '')
        data = data.split('\n')

        # Parse out unique fields
        time = data[0].split(' ')[0]
        headers.append(time)
        data = data[1:] # remove time

        for datum in data:
            headers.append(datum.split(' = ')[0])

        return headers

    def parse_config_data(self, configFile, ss_breakdown):
        parsed_configs = extract_configs(self.metadata_file_path, [configFile])
        if ss_breakdown == False:
            num_elements = len(parsed_configs['subsystem_assignments'])
            parsed_configs['subsystem_assignments'] = [['MISSION'] for elem in range(num_elements)]
        return parsed_configs

    ##### GETTERS ##################################
    def get_sim_data(self):
        return self.all_headers, self.sim_data, self.binning_configs

