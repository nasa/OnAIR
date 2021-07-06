"""
42 Parser
"""

import csv
import time
import os
import re
import time

from src.util.file_io import *
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
                        
            self.all_headers = headers_dict
            self.sim_data = data_dict

            # Setup binning config information
            self.binning_configs = {}
            self.binning_configs['subsystem_assignments'] = {}
            self.binning_configs['test_assignments'] = {}
            self.binning_configs['description_assignments'] = {}
            
            for config_file in str2lst(configFiles):
                # Config format {'subsystem' : {data_file : ['Etc.']}}
                config = self.parse_config_data(config_file, ss_breakdown)
                # Although this is a for loop, it should only execute once because the config currently has only read in one single file 
                for data_file_key in config['subsystem_assignments']:
                    self.binning_configs['subsystem_assignments'][data_file_key] = config['subsystem_assignments'][data_file_key]
                    self.binning_configs['test_assignments'][data_file_key] = config['test_assignments'][data_file_key]
                    self.binning_configs['description_assignments'][data_file_key] = config['description_assignments'][data_file_key]

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

        for dataPt in dataPts:
            frames.append(self.parse_frame(dataPt))

        # Process into binning format
        all_headers = {}
        all_data = {}

        times = [frame[0] for frame in frames]
        all_headers[dataFile] = headers

        for i in range(len(times)):
            all_data[times[i]] = {dataFile : frames[i]}

        return all_headers, all_data

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

    def parse_frame(self, frame):
        clean_frame = []
        data = frame.split('\n')

        # Parse out unique fields
        time = data[0].split(' ')[1]
        clean_frame.append(time)
        data = data[1:] # remove time

        spacecraft = data[0].split('.')[0] # Not doing anything with this for now.. may parse it out

        for datum in data:
            clean_frame.append(datum.split(' = ')[1])

        return clean_frame

    def parse_config_data(self, configFile, ss_breakdown):
        parsed_configs = extract_configs(self.metadata_file_path, [configFile])
        if ss_breakdown == False:
            num_elements = len(parsed_configs['subsystem_assignments'][process_filepath(configFile)])
            parsed_configs['subsystem_assignments'][process_filepath(configFile)] = [['MISSION'] for elem in range(num_elements)]
        return parsed_configs

    ##### GETTERS ##################################
    def get_sim_data(self):
        return self.all_headers, self.sim_data, self.binning_configs

