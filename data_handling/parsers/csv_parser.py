"""
CSV Parser
"""

import os
import pandas as pd

from data_handling.parsers.on_air_parser import OnAirParser
from src.util.print_io import *
from data_handling.parsers.parser_util import * 

class CSV(OnAirParser):
    def pre_process_data(self, dataFiles):
        pass

    def process_data_per_data_file(self, data_file):
        labels, data = self.parse_csv_data(data_file)
        # Header format : { Filename : ['Header', 'Another Header', 'Etc.']}
        self.all_headers[data_file] = labels[data_file] #The key for labels is the file name, so we're able to add that to our "big" dictionary
        # Data format : { 'index': { Filename : ['Data_Point', 'Another Data_Point', 'Etc.']}}
        for key in data:
            # If the key does not exist in the data dictionary already, then data_dict[key][data_file] will give an error
            # In order to skirt this issue, if the key does not exist create an empty dictionary for it, then give it the data file
            if key in self.sim_data:
                self.sim_data[key][data_file] = data[key][data_file]
            else:
                self.sim_data[key] = {}
                self.sim_data[key][data_file] = data[key][data_file]
            

##### INITIAL PROCESSING ####
    def parse_csv_data(self, dataFile):
        #Read in the data set
        dataset = pd.read_csv(os.path.join(self.raw_data_filepath, dataFile), delimiter=',', header=0, dtype=str)
        dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]
        #Get headers from dataset and format them into a dictionary
        # {fileName: [Header1, Header2, ...]}
        headersDict = {}
        headersDict[dataFile] = list(dataset.columns.values)
        all_headers = headersDict
        #Find the 'Time' header in the list in order to match 42 file formatting 
        # Converting
        upperCaseStringHeaders = [x.upper().strip() for x in headersDict[dataFile] if isinstance(x, str)]
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

    def parse_config_data(self, configFile, ss_breakdown):
        parsed_configs = extract_configs(self.metadata_filepath, [configFile], csv=True)
        if ss_breakdown == False:
            num_elements = len(parsed_configs['subsystem_assignments'])
            parsed_configs['subsystem_assignments'] = [['MISSION'] for elem in range(num_elements)]
        return parsed_configs

##### GETTERS ##################################

    def get_sim_data(self):
        return self.all_headers, self.sim_data, self.binning_configs
    