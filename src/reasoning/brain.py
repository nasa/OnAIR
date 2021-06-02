"""
Brain Class
Deals with supervised learning for diagnosing statuses
"""

import csv
import copy 

from src.reasoning.status import Status
from src.reasoning.diagnosis import Diagnosis
from src.DS.pyds.pyds import MassFunction as DSFrame
from src.data_driven_components.supervised_learning import SupervisedLearning
from src.data_driven_components.associativity import Associativity
from src.subsystems.spacecraft import Spacecraft
from src.util.print_io import *

class Brain:
    def __init__(self, spacecraft):
        self.init_conditions = copy.deepcopy(spacecraft)
        self.spacecraft_rep = spacecraft
        self.mission_status = self.spacecraft_rep.get_status()
        self.bayesian_status = self.spacecraft_rep.get_bayesian_status()
        self.DS_info = self.spacecraft_rep.get_DS_status()
        self.DS_status = self.DS_info[0]
        self.interpreted_status = self.calc_status()
        self.supervised_learning = SupervisedLearning(self.spacecraft_rep.get_current_data())

    # Markov Assumption holds 
    def reason(self, frame):
        self.spacecraft_rep.update(frame)
        self.mission_status = self.spacecraft_rep.get_status() #Nominal Reasoning, just feasibility to start
        self.DS_info = self.spacecraft_rep.get_DS_status()
        self.DS_status = self.DS_info[0]
        self.supervised_learning.update(self.spacecraft_rep.get_current_data(True), self.mission_status)
        self.NN_status = self.supervised_learning.predict(self.spacecraft_rep.get_current_data(True))
        self.interpreted_status = self.calc_status()  # interpreted status, given info

    def diagnose(self, time_step):
        lstm_predict = self.supervised_learning.lstm_predict()
        tree_traversal = self.spacecraft_rep.get_status_object().fault_traversal()
        faults = self.spacecraft_rep.get_faulting_mnemonics()
        activations = self.supervised_learning.associations.get_current_activations()
        importance_sampling = self.supervised_learning.get_importance_sampling()
        graph = self.supervised_learning.get_associativity_graph()
        associativity_list = [(elem['name'], elem['statuses'][0]) for elem in self.get_sensor_status_report()]
        graph.set_header_statuses(associativity_list)
        benchmark_graph = self.supervised_learning.get_benchmark_graph()
        associativity_metrics = self.supervised_learning.get_associativity_metrics()

        diagnosis = Diagnosis()
        diagnosis.set_diagnosis_val('time_step', time_step)
        diagnosis.set_diagnosis_val('mission_status', self.mission_status)
        diagnosis.set_diagnosis_val('bayesian_status', self.bayesian_status)
        diagnosis.set_diagnosis_val('DS_info', self.DS_info)
        diagnosis.set_diagnosis_val('DS_status', self.DS_status)
        diagnosis.set_diagnosis_val('NN_prediction', self.NN_status)
        diagnosis.set_diagnosis_val('LSTM_prediction', lstm_predict)
        diagnosis.set_diagnosis_val('interpreted_status', self.interpreted_status)
        diagnosis.set_diagnosis_val('fault_tree', tree_traversal)
        diagnosis.set_diagnosis_val('current_activations', activations)
        diagnosis.set_diagnosis_val('associativity_graph', graph)
        diagnosis.set_diagnosis_val('importance_sampling', importance_sampling)
        diagnosis.set_diagnosis_val('benchmark_graph', benchmark_graph)
        diagnosis.set_diagnosis_val('associativity_metrics', associativity_metrics)
        diagnosis.perform_diagnosis()
        return diagnosis

    def calc_status(self):
        if self.mission_status == self.DS_status:
            return self.mission_status
        else:
            if self.DS_info[1] < 0.5:
                return self.mission_status
            elif self.DS_info[2] > 0.1:
                return self.mission_status
            else:
                return self.DS_info[0] # Need a way to reason between DS and mission status when they conflict.
                                       # That is, benign safe modes specifically
                                       
    def set_past_history(self, simData):
        sc = copy.deepcopy(self.init_conditions)
        input_samples = []
        output_samples = []
        while simData.has_more():
            frame = simData.get_next()
            sc.update(frame)
            input_samples.append(sc.get_current_data(True))
            output_samples.append(sc.get_status_object().status_report_hash_list())
        self.supervised_learning.set_historical_data(input_samples, output_samples)

    def get_hierarchical_status_report(self):
        return self.spacecraft_rep.get_status_object().status_report_traversal()

    def get_sensor_status_report(self):
        return self.spacecraft_rep.get_status_object().sensor_status_report_hash_list()

