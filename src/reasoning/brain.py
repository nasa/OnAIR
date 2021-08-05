"""
Brain Class
Deals with supervised learning for diagnosing statuses
"""

import csv
import copy 

from src.systems.status import Status
from src.reasoning.diagnosis import Diagnosis
from src.data_driven_components.data_driven_learning import DataDrivenLearning
from src.systems.spacecraft import Spacecraft
# from src.util.print_io import *

class Brain:
    def __init__(self, spacecraft=None):
        self.spacecraft_rep = spacecraft
        self.learning_systems = DataDrivenLearning(self.spacecraft_rep.get_headers())
        self.mission_status = self.spacecraft_rep.get_status()
        self.bayesian_status = self.spacecraft_rep.get_bayesian_status()

    # Markov Assumption holds 
    def reason(self, frame):
        self.spacecraft_rep.update(frame)
        self.mission_status = self.spacecraft_rep.get_status() 
        self.bayesian_status = self.spacecraft_rep.get_bayesian_status() 
        self.learning_systems.update(frame, self.mission_status)

    def diagnose(self, time_step):
        learning_systems_results = self.learning_systems.diagnose(self.spacecraft_rep.get_current_faulting_mnemonics())
        diagnosis = Diagnosis(time_step, 
                              learning_systems_results, 
                              self.bayesian_status,
                              self.spacecraft_rep.get_current_faulting_mnemonics())
        return diagnosis





