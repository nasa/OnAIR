"""
Brain Class
Deals with supervised learning for diagnosing statuses
"""

import csv
import copy 

from src.systems.status import Status
from src.data_driven_components.data_driven_learning import DataDrivenLearning
from src.systems.spacecraft import VehicleRepresentation
from src.reasoning.diagnosis import Diagnosis

class Brain:
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

