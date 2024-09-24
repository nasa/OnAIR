# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

"""
Status Class
"""


class Status:
    def __init__(self, name="MISSION", stat="---", conf=-1.0):
        self.name = name
        self.set_status(stat, conf)

    ##### GETTERS & SETTERS ##################################
    def set_status(self, stat, bayesianConf=-1.0):
        assert -1.0 <= bayesianConf <= 1.0
        assert stat in ["---", "RED", "YELLOW", "GREEN"]
        self.status = stat
        self.bayesian_conf = bayesianConf

    def get_status(self):
        return self.status

    def get_bayesian_status(self):
        return (self.status, self.bayesian_conf)

    def get_name(self):
        return self.name
