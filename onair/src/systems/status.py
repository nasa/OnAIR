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
    """
    Class representing the status of a telemetry data field.

    Args:
        name (str): Name of the telemetry data field.
        stat (str): Initial status ('---', 'RED', 'YELLOW', 'GREEN').
        conf (float): Initial Bayesian confidence score (-1.0 to 1.0).

    Attributes:
        name (str): Name of the telemetry data field.
        status (str): Current status ('---', 'RED', 'YELLOW', 'GREEN').
        bayesian_conf (float): Current Bayesian confidence score (-1.0 to 1.0).
    """
    def __init__(self, name='MISSION', stat='---', conf=-1.0):
        self.name =  name
        self.set_status(stat, conf)

    ##### GETTERS & SETTERS ##################################
    def set_status(self, stat, bayesianConf=-1.0):
        assert(-1.0 <= bayesianConf <= 1.0)
        assert(stat in ['---', 'RED', 'YELLOW', 'GREEN'])
        self.status = stat
        self.bayesian_conf = bayesianConf

    def get_status(self):
        return self.status

    def get_bayesian_status(self):
        """
        Get the current status and Bayesian confidence score.

        Returns:
            tuple: A tuple containing the current status and Bayesian confidence score.
        """
        return (self.status, self.bayesian_conf)

    def get_name(self):
        """
        Get the name of the telemetry data field.

        Returns:
            str: Name of the telemetry data field.
        """
        return self.name
