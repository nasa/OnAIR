# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

"""
Agent Class
Deals with supervised learning for diagnosing statuses
"""
from ..data_driven_components.data_driven_learning import DataDrivenLearning
from ..reasoning.diagnosis import Diagnosis

class Agent:
    def __init__(self, vehicle):
        self.vehicle_rep = vehicle
        self.learning_systems = DataDrivenLearning(self.vehicle_rep.get_headers())
        self.mission_status = self.vehicle_rep.get_status()
        self.bayesian_status = self.vehicle_rep.get_bayesian_status()

    # Markov Assumption holds 
    def reason(self, frame):
        self.vehicle_rep.update(frame)
        self.mission_status = self.vehicle_rep.get_status() 
        self.learning_systems.update(frame, self.mission_status)

    def diagnose(self, time_step):
        """ Grab the mnemonics from the """
        learning_system_results = self.learning_systems.render_diagnosis() 
        diagnosis = Diagnosis(time_step, 
                              learning_system_results,
                              self.bayesian_status,
                              self.vehicle_rep.get_current_faulting_mnemonics())
        return diagnosis.perform_diagnosis()

