"""
TimeSynchronizer class
Helper class used by driver.py to process and organize data
"""

import time
import ast
import copy

from collections import OrderedDict 

from src.data_handling.parsers.parser_util import str2lst

class TimeSynchronizer:
    def __init__(self, headers={}, dataFrames={}, test_configs={}):
        try:
            # Both have side effects
            self.init_sync_data(headers, test_configs) 
            self.sort_data(dataFrames)
        except:
            self.ordered_sources = []
            self.ordered_fused_headers = []
            self.ordered_fused_tests = []
            self.indices_to_remove = []
            self.offsets = {}
            self.sim_data =  []

    def init_sync_data(self, headers, configs):
        if headers == {} or configs == {}:
            raise Exception('Unable to initialize sync data: Empty Dataset')

        self.ordered_sources = list(headers.keys())
        unclean_fused_hdrs = [] # Needed for sorting data
        unclean_fused_tests = [] # Needed for sorting data
        src_start_indices = {} # Needed for sorting data

        i = 0
        for source_file in self.ordered_sources:
            unclean_fused_hdrs.extend(headers[source_file])
            unclean_fused_tests.extend(configs['test_assignments'][source_file])

            src_start_indices[source_file] = i
            i = i + len(headers[source_file])
            
        time_indices_for_removal, self.ordered_fused_headers = self.remove_time_headers(copy.deepcopy(unclean_fused_hdrs))
        
        self.ordered_fused_tests = self.remove_time_datapoints(unclean_fused_tests, time_indices_for_removal)
        self.ordered_fused_headers.insert(0, 'TIME')
        self.ordered_fused_tests.insert(0, [['SYNC', 'TIME']])

        self.indices_to_remove = time_indices_for_removal
        self.offsets = src_start_indices

    def sort_data(self, dataFrames):
        sorted_data = []
        num_sources = len(self.ordered_sources)
        num_frames = len(dataFrames.keys())

        total_times = list(dataFrames.keys())
        total_times.sort()

        sorted_data = []

        for time in total_times:
            dataFromMultSources = dataFrames[time] # can be from any data source
            num_unclean_hdrs = len(self.indices_to_remove) + len(self.ordered_fused_headers) - 1 # unclean list, remove one of the added 'TIME' in clean
            clean_array_of_data = ['-']*num_unclean_hdrs
            for source in dataFrames[time].keys():
                index_offset = self.offsets[source]
                data = dataFrames[time][source]
                for datum_index in range(len(data)):
                    clean_array_of_data[datum_index + index_offset] = data[datum_index]

            self.remove_time_datapoints(clean_array_of_data, copy.deepcopy(self.indices_to_remove))
            clean_array_of_data.insert(0, time)
            sorted_data.append(clean_array_of_data)

        self.sim_data = sorted_data


    def remove_time_headers(self, hdrs_list):
        indices = []
        variations = ['TIME', 'time']

        for j in range(len(hdrs_list)):
            if hdrs_list[j] in variations:
                indices.append(j)

        clean_hdrs_list = copy.deepcopy(hdrs_list)
            
        num_removed = 0
        for i in range(len(indices)):
            curr_index = indices[i]
            del clean_hdrs_list[curr_index-num_removed]
            num_removed = num_removed + 1

        return indices, clean_hdrs_list


    def remove_time_datapoints(self, data, indices_to_remove):
        outstanding_removal = indices_to_remove
        num_removed = 0 

        for i in range(len(outstanding_removal)):
            curr_index = outstanding_removal[i]
            del data[curr_index-num_removed]
            num_removed = num_removed + 1
        return data

    def get_spacecraft_metadata(self):
        return self.ordered_fused_headers, self.ordered_fused_tests

