"""
Spacecraft Class
Handles retrieval and storage of spacecraft subsystem information
"""

from src.systems.status import Status
from src.util.print_io import *

# from src.util.tm_tests_util import * 
# from src.util.telemetry_test_suite import TelemetryTestSuite

class Spacecraft:
    def __init__(self, headers=[], tests=[]): # metaData is a timesynchronizer obj
        try:
            self.init_spacecraft(headers, tests)
        except:
            self.status = Status('MISSION')
            
            self.headers = []
            self.tests = []
            self.curr_data = []

    def init_spacecraft(self, headers, tests):
        assert(len(headers) == len(tests))
        self.status = Status('MISSION')
        self.headers = headers
        self.tests = tests
        self.curr_data = ['-']* len(self.headers)

    ##### UPDATERS #################################
    def update(self, frame):        
        for i in range(len(frame)):
            if frame[i] != '-':
                self.curr_data[i] = frame[i]

        # UPDATE STATUS! 

    ##### GETTERS AND SETTERS #####
    def get_current_data(self):
        return self.curr_data

    def get_current_time(self):
        return self.curr_data[0]

    def get_status(self):
        return self.status.get_status()

    def get_bayesian_status(self):
        return self.status.get_bayesian_status()



        #     def update(self, dataframe, time='NA'):
        #         headers_to_update = dataframe['headers']
        #         raw_data = dataframe['data']

        #         for i in range(len(headers_to_update)):
        #             self.data[headers_to_update[i]] = raw_data[i]

        #         num_fields = len(headers_to_update)
        #         assert(num_fields == len(raw_data))

        #         if time != 'NA':
        #             # self.mission_time = time
        #             self.sync_data['TIME'] = time

        #         self.test_suite.execute(dataframe, self.sync_data)

        #         for i in range(0, num_fields):
        #             field_name = headers_to_update[i]
        #             test_results = self.test_suite.get_latest_result(field_name)
        #             args = [field_name, test_results.get_stat(), test_results.get_bayesian_conf(), test_results.get_DS_Frame()] # Update with Boe? fuse? 
        #             self.status.update(args) # update each field 

    # def get_current_mission_data_list(self, mission_data):
    #     report = {'headers' : self.headers}
    #     mission_data_list = []
    #     for hdr in self.headers:
    #         mission_data_list.append(mission_data[hdr])
    #     report['data'] = mission_data_list
    #     return report

    # def get_current_data(self, lst_format=False):
    #     all_systems_dict = {}
    #     all_systems_list = {}
    #     mission = {}

    #     for ss in self.get_subsystems():
    #         all_systems_dict[ss.get_name()] = ss.get_current_data()
    #         all_systems_list[ss.get_name()] = ss.get_current_data(True)

    #         ss_data = ss.get_current_data()
    #         for sensor in ss_data.keys():
    #             mission[sensor] = ss_data[sensor]

    #     all_systems_list['MISSION'] = self.get_current_mission_data_list(mission)
    #     all_systems_dict['MISSION'] = mission        
        
    #     if lst_format == True: 
    #         return all_systems_list
    #     return all_systems_dict

    # def get_status_object(self):
    #     return self.status

    # def get_status(self):
    #     return self.status.get_status()

    # def get_bayesian_status(self):
    #     return self.status.get_bayesian_status()

    # def get_DS_status(self):
    #     return self.status.get_DS_status()

    # def get_faulting_mnemonics(self):
    #     tree_traversal = self.get_status_object().fault_traversal()
    #     faults = list(set([path[-1] for path in tree_traversal]))
    #     return faults

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# """
# Subsystem Class
# Parent for Spacecraft subsystems, including: CDH, Communication, Electrical, GNC, Power, Propulsion, and Thermal
# """

# from src.util.print_io import * 
# from src.util.tm_tests_util import * 
# from src.util.telemetry_test_suite import TelemetryTestSuite
# from src.reasoning.status import Status

# class SubSystem:
#     def __init__(self, metaData, name='OVERALL'):
#         self.type = name

#         self.headers = [elem[0] for elem in metaData]
#         tests = {}
#         data = {}
        
#         for tup in metaData:
#             tests[tup[0]] = tup[1]
#             data[tup[0]] = 0.0
        
#         self.data = data       
#         self.test_suite = TelemetryTestSuite(self.type, self.headers, tests) 
#         self.status = Status(self.type)
#         self.status.init_ss_list(self.headers)
#         self.sync_data = {'TIME' : 'NA',
#                              'X' : 0.0,
#                              'Y' : 0.0,
#                              'Z' : 0.0}
#         self.epsilon = 0.1 # percentage of range to test. If range is 0,10, it would be 10%
#         self.uncertainty = 0.0

#     def update(self, dataframe, time='NA'):
#         headers_to_update = dataframe['headers']
#         raw_data = dataframe['data']

#         for i in range(len(headers_to_update)):
#             self.data[headers_to_update[i]] = raw_data[i]

#         num_fields = len(headers_to_update)
#         assert(num_fields == len(raw_data))

#         if time != 'NA':
#             # self.mission_time = time
#             self.sync_data['TIME'] = time

#         self.test_suite.execute(dataframe, self.sync_data)

#         for i in range(0, num_fields):
#             field_name = headers_to_update[i]
#             test_results = self.test_suite.get_latest_result(field_name)
#             args = [field_name, test_results.get_stat(), test_results.get_bayesian_conf(), test_results.get_DS_Frame()] # Update with Boe? fuse? 
#             self.status.update(args) # update each field 

#     ##### GETTERS AND SETTERS #####
#     def get_status_object(self):
#         return self.status

#     def get_status(self):
#         return self.status.get_status()

#     def get_headers(self):
#         return self.headers

#     def get_current_data(self, lst_format=False):
#         if lst_format == True:
#             return self.get_current_data_list() 
#         return self.data

#     def get_current_data_list(self):
#         report = {'headers' : self.headers}
#         curr_data = [self.data[hdr] for hdr in self.headers]
#         report['data'] = curr_data
#         return report

#     def get_name(self):
#         return self.type

#     ##### I/O #####
#     def __str__(self):
#         return subsystem_status_str(self)


