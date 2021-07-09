"""
Diagnosis Class, used to store all diagnosis results / summary info 
"""

import csv
import copy 

class Diagnosis:
    def __init__(self, time_step=0, 
                       learning_systems_results = {'vae_diagnosis' : None,
                                                   'associativity_diagnosis' : None,
                                                   'pomdp_diagnosis' : None},
                       status_confidence=1.0, 
                       faulting_mnemonics=[]):

        self.time_step = time_step
        self.vae_results = learning_systems_results['vae_diagnosis']
        self.assoc_results = learning_systems_results['associativity_diagnosis']
        self.pomdp_results = learning_systems_results['pomdp_diagnosis']

        self.status_confidence = status_confidence
        self.faulting_mnemonics = faulting_mnemonics

    def perform_diagnosis(self):
        return

    def results_csv(self):
        return

    def cohensKappa(self, TP, FN, FP, TN, allValues=False):
        return -1.0

    def __str__(self):
        result = ''
        result = result + '\nTime Step:                ' + str(self.time_step) + '\n'
        result = result + 'RED Status Confidence:    ' + str(self.status_confidence) + '\n'
        result = result + 'Faulting Mnemonics:       ' + ',' .join(str(s) for s in self.faulting_mnemonics) + '\n'
        result = result + 'VAE Diagnosis:            ' + str(self.vae_results) + '\n'
        result = result + 'Associativity Diagnosis:  ' + str(self.assoc_results) + '\n'
        result = result + 'POMDP Diagnosis:          ' + str(self.pomdp_results) + '\n'
        return result

