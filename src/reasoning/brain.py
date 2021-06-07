"""
Brain Class
Deals with supervised learning for diagnosing statuses
"""

import csv
import copy 

from src.systems.status import Status
# from src.reasoning.diagnosis import Diagnosis
from src.data_driven_components.data_driven_learning import DataDrivenLearning
# from src.data_driven_components.associativity import Associativity
from src.systems.spacecraft import Spacecraft
# from src.util.print_io import *

class Brain:
    def __init__(self, spacecraft=None):
        try:
            self.init_brain(spacecraft)
        except:
            self.spacecraft_rep = None
            self.learning_systems = None
            self.mission_status = '---'
            self.bayesian_status = -1.0

    def init_brain(self, spacecraft):
        self.spacecraft_rep = spacecraft
        self.learning_systems = DataDrivenLearning(self.spacecraft_rep.get_headers())
        self.mission_status = self.spacecraft_rep.get_status()
        self.bayesian_status = self.spacecraft_rep.get_bayesian_status()

    # Markov Assumption holds 
    def reason(self, frame):
        self.spacecraft_rep.update(frame)
        self.mission_status = self.spacecraft_rep.get_status() 
        self.learning_systems.update(frame, self.mission_status)

    def diagnose(self, time_step):
        return None
    #     lstm_predict = self.supervised_learning.lstm_predict()
    #     tree_traversal = self.spacecraft_rep.get_status_object().fault_traversal()
    #     faults = self.spacecraft_rep.get_faulting_mnemonics()
    #     activations = self.supervised_learning.associations.get_current_activations()
    #     importance_sampling = self.supervised_learning.get_importance_sampling()
    #     graph = self.supervised_learning.get_associativity_graph()
    #     associativity_list = [(elem['name'], elem['statuses'][0]) for elem in self.get_sensor_status_report()]
    #     graph.set_header_statuses(associativity_list)
    #     benchmark_graph = self.supervised_learning.get_benchmark_graph()
    #     associativity_metrics = self.supervised_learning.get_associativity_metrics()

    #     diagnosis = Diagnosis()
    #     diagnosis.set_diagnosis_val('time_step', time_step)
    #     diagnosis.set_diagnosis_val('mission_status', self.mission_status)
    #     diagnosis.set_diagnosis_val('bayesian_status', self.bayesian_status)
    #     diagnosis.set_diagnosis_val('DS_info', self.DS_info)
    #     diagnosis.set_diagnosis_val('DS_status', self.DS_status)
    #     diagnosis.set_diagnosis_val('NN_prediction', self.NN_status)
    #     diagnosis.set_diagnosis_val('LSTM_prediction', lstm_predict)
    #     diagnosis.set_diagnosis_val('interpreted_status', self.interpreted_status)
    #     diagnosis.set_diagnosis_val('fault_tree', tree_traversal)
    #     diagnosis.set_diagnosis_val('current_activations', activations)
    #     diagnosis.set_diagnosis_val('associativity_graph', graph)
    #     diagnosis.set_diagnosis_val('importance_sampling', importance_sampling)
    #     diagnosis.set_diagnosis_val('benchmark_graph', benchmark_graph)
    #     diagnosis.set_diagnosis_val('associativity_metrics', associativity_metrics)
    #     diagnosis.perform_diagnosis()
    #     return diagnosis

                                       
    # def set_past_history(self, simData):
    #     sc = copy.deepcopy(self.init_conditions)
    #     input_samples = []
    #     output_samples = []
    #     while simData.has_more():
    #         frame = simData.get_next()
    #         sc.update(frame)
    #         input_samples.append(sc.get_current_data(True))
    #         output_samples.append(sc.get_status_object().status_report_hash_list())
    #     self.supervised_learning.set_historical_data(input_samples, output_samples)

    # def get_sensor_status_report(self):
    #     return self.spacecraft_rep.get_status_object().sensor_status_report_hash_list()

