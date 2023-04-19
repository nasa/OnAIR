"""
TimeSynchronizer class
Helper class used by driver.py to process and organize data
"""

import time
import ast
import copy

from collections import OrderedDict 

from data_handling.parsers.parser_util import str2lst

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
        
        total_times = list(dataFrames.keys())
        total_times.sort()

        sorted_data = []

        for time in total_times:
            clean_data_array = self.initialize_clean_data_array(dataFrames, time)
            sorted_data.append(clean_data_array)

        self.sim_data = sorted_data

    def initialize_clean_data_array(self, dataFrames, time):
        num_unclean_hdrs = len(self.indices_to_remove) + len(self.ordered_fused_headers) - 1 # unclean list, remove one of the added 'TIME' in clean
        clean_data_array = ['-']*num_unclean_hdrs
        for source in dataFrames[time].keys():
            index_offset = self.offsets[source]
            data = dataFrames[time][source]
            self.copy_to_with_offset(clean_data_array, data, index_offset)

        self.remove_time_datapoints(clean_data_array, copy.deepcopy(self.indices_to_remove))
        clean_data_array.insert(0, time)

        return clean_data_array
    
    def copy_to_with_offset(self, dest_array, src_array, offset):
        for datum_index in range(len(src_array)):
            dest_array[datum_index + offset] = src_array[datum_index]

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
        outstanding_removal = indices_to_remove.copy()
        outstanding_removal.sort(reverse=True)

        for i in outstanding_removal:
            del data[i]

        return data

    def get_vehicle_metadata(self):
        return self.ordered_fused_headers, self.ordered_fused_tests

    def get_sim_data(self):
        return self.sim_data
