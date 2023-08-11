# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

"""
VehicleRepresentation Class
Handles retrieval and storage of vehicle subsystem information
"""

from .status import Status
from .telemetry_test_suite import TelemetryTestSuite

from ..util.print_io import *

class VehicleRepresentation:
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
        self.status.set_status(*self.test_suite.get_suite_status())

    ##### GETTERS AND SETTERS #####
    def get_headers(self):
        return self.headers

    def get_current_faulting_mnemonics(self):
        return self.test_suite.get_status_specific_mnemonics()

    def get_current_data(self):
        return self.curr_data

    def get_current_time(self):
        return self.curr_data[0]

    def get_status(self):
        return self.status.get_status()

    def get_bayesian_status(self):
        return self.status.get_bayesian_status()

    def get_batch_status_reports(self, batch_data):
        return


