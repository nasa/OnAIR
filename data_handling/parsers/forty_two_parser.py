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
from data_handling.parsers.parser_util import * 

class FortyTwo:
    def __init__(self, rawDataFilepath = '', 
                      metadataFilepath = '', 
                             dataFiles = '', 
                           configFiles = '', 
                          ss_breakdown = False):
        """An initial parsing needs to happen in order to use the parser classes
           This means that, if you want to use this class to parse in real time, 
           it needs to at least have seen one sample of the anticipated format """
        
        self.raw_data_filepath = rawDataFilepath
        self.metadata_filepath = metadataFilepath
        self.all_headers = ''
        self.sim_data = ''
        self.binning_configs = ''

        if (dataFiles != '') and (configFiles != ''):
            labels, data = self.parse_sim_data(str2lst(dataFiles)[0])
            self.all_headers = labels
            self.sim_data = data

            configs = self.parse_config_data(str2lst(configFiles)[0], ss_breakdown)

            self.binning_configs = {}
            self.binning_configs['subsystem_assignments'] = {}
            self.binning_configs['test_assignments'] = {}
            self.binning_configs['description_assignments'] = {}

            for data_file in str2lst(dataFiles):
                self.binning_configs['subsystem_assignments'][data_file] = configs['subsystem_assignments']
                self.binning_configs['test_assignments'][data_file] = configs['test_assignments']
                self.binning_configs['description_assignments'][data_file] = configs['description_assignments']

    ##### INITIAL PROCESSING ####
    def parse_sim_data(self, dataFile):
        headers = []
        frames = []

        txt_file = open(self.raw_data_filepath + dataFile,"r+")

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

        vehicle = data[0].split('.')[0] # Not doing anything with this for now.. may parse it out

        for datum in data:
            clean_frame.append(datum.split(' = ')[1])

        return clean_frame

    def parse_config_data(self, configFile, ss_breakdown):
        parsed_configs = extract_configs(self.metadata_filepath, configFile)
        if ss_breakdown == False:
            num_elements = len(parsed_configs['subsystem_assignments'])
            parsed_configs['subsystem_assignments'] = [['MISSION'] for elem in range(num_elements)]
        return parsed_configs

    ##### GETTERS ##################################
    def get_sim_data(self):
        return self.all_headers, self.sim_data, self.binning_configs

