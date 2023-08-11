# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

from abc import ABC, abstractmethod
from .parser_util import * 

class OnAirParser(ABC):
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
      self.all_headers = {}
      self.sim_data = {}
      self.binning_configs = {}

      if (dataFiles != '') and (configFiles != ''):
          self.pre_process_data(dataFiles)
  
          self.binning_configs['subsystem_assignments'] = {}
          self.binning_configs['test_assignments'] = {}
          self.binning_configs['description_assignments'] = {}

          configs = self.parse_config_data(str2lst(configFiles)[0], ss_breakdown)

          for data_file in str2lst(dataFiles):
              self.process_data_per_data_file(data_file)
              self.binning_configs['subsystem_assignments'][data_file] = configs['subsystem_assignments']
              self.binning_configs['test_assignments'][data_file] = configs['test_assignments']
              self.binning_configs['description_assignments'][data_file] = configs['description_assignments']

    @abstractmethod
    def pre_process_data(self, dataFiles):
        """
        Adjust dataFiles before parsing into binning_configs
        """
        raise NotImplementedError

    @abstractmethod
    def process_data_per_data_file(self, data_file):
        """
        Adjust each individual data_file before parsing into binning_configs
        """
        raise NotImplementedError

    @abstractmethod
    def parse_config_data(self, configFile, ss_breakdown):
        """
        Create the configs that will be used to populate the binning_configs for the data files
        """
        raise NotImplementedError
