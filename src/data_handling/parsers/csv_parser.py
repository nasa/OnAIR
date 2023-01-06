"""
CSV Parser
"""
import os

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

            #Get labels and data
            # Data format: {index(hopefully synced):{filename : [Attribute value1, attribute value2, ...]}}
            labels, data = self.parse_csv_data(str2lst(dataFiles)[0])
            self.all_headers = labels              
            self.sim_data = data

            #Get CSV Configuration
            configs = self.parse_config_data_CSV(str2lst(configFiles)[0], ss_breakdown)

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
    