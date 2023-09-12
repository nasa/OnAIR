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
    def __init__(self, data_file, meta_file, ss_breakdown = False):
        """An initial parsing needs to happen in order to use the parser classes
            This means that, if you want to use this class to parse in real time,
            it needs to at least have seen one sample of the anticipated format """

        self.raw_data_file = data_file
        self.meta_data_file = meta_file
        self.all_headers = {}
        self.sim_data = {}
        self.binning_configs = {}

        configs = self.parse_meta_data_file(self.meta_data_file, ss_breakdown)
        self.binning_configs['subsystem_assignments'] = configs['subsystem_assignments']
        self.binning_configs['test_assignments'] = configs['test_assignments']
        self.binning_configs['description_assignments'] = configs['description_assignments']

        self.process_data_file(self.raw_data_file)

    @abstractmethod
    def process_data_file(self, data_file):
        """
        Adjust each individual data_file before parsing into binning_configs
        """
        raise NotImplementedError

    @abstractmethod
    def parse_meta_data_file(self, meta_data_file, ss_breakdown):
        """
        Create the configs that will be used to populate the binning_configs for the data files
        """
        raise NotImplementedError
