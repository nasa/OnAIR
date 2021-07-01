"""
Spacecraft Class
Handles retrieval and storage of spacecraft subsystem information
"""

from src.systems.status import Status
from src.systems.telemetry_test_suite import TelemetryTestSuite

from src.util.print_io import *

class Spacecraft:
    def __init__(self, headers=[], tests=[]): # metaData is a timesynchronizer obj
        assert(len(headers) == len(tests))
        self.status = Status('MISSION')
        self.headers = headers
        self.test_suite = TelemetryTestSuite(headers, tests)
        self.curr_data = ['-']* len(self.headers)

    ##### UPDATERS #################################
    def update(self, frame):
        for i in range(len(frame)):
            if frame[i] != '-':
                self.curr_data[i] = frame[i]
        self.test_suite.execute_suite(frame)
        self.mission_status = self.test_suite.get_suite_status()

        # UPDATE STATUS! 

    ##### GETTERS AND SETTERS #####
    def get_headers(self):
        return self.headers

    def get_current_data(self):
        return self.curr_data

    def get_current_time(self):
        return self.curr_data[0]

    def get_status(self):
        return self.status.get_status()

    def get_bayesian_status(self):
        return self.status.get_bayesian_status()



