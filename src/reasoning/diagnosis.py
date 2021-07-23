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
                       faulting_mnemonics=[],
                       ground_truth = None):

        self.time_step = time_step
        self.vae_results = learning_systems_results['vae_diagnosis']
        self.assoc_results = learning_systems_results['associativity_diagnosis']
        self.pomdp_results = learning_systems_results['pomdp_diagnosis']

        self.status_confidence = status_confidence
        self.faulting_mnemonics = faulting_mnemonics
        self.ground_truth = ground_truth 

    def perform_diagnosis(self):
        correct_diagnoses_VAE = False

        # Should do one for EACH value, and show how many it got. % wise 
        if (all(x in self.vae_results[0:3] for x in self.ground_truth[0])):
        # if self.ground_truth[0][0] in list(self.vae_results[0:3]):
            print(f'Main VAE diagnoses agrees with groundtruth')
            correct_diagnoses_VAE = True
        # else:
        #     print(f'Main groundtruth {self.ground_truth} disagrees with main VAE diagnoses {final_diagnosis.vae_results[0]}')
        
        return correct_diagnoses_VAE


    def cumulative_topK_accuracy_VAE(self, k):
        """
        Given max k returns array where the i-th element is top-i accuracy. Accuracy is defined as
            whether those top-i guesses contain of of the groundtruth elements
        :param k: (int) maximum top-k
        :returns: (numpy array) length k array where i-th element is top-i accuracy
        """
        acc = np.zeros(k)

        for i in range(k):
            for g in self.ground_truth[0]:
                if g in self.vae_results[:i]:
                    acc[i] += 1
        return acc

    def results_csv(self):
        return

    def cohensKappa(self, TP, FN, FP, TN, allValues=False):
        return -1.0

    def set_ground_truth(self, gt):
        self.ground_truth = gt

    def __str__(self):
        result = ''
        result = result + '\nTime Step:                ' + str(self.time_step) + '\n'
        result = result + 'RED Status Confidence:    ' + str(self.status_confidence) + '\n'
        result = result + 'Faulting Mnemonics:       ' + ',' .join(str(s) for s in self.faulting_mnemonics) + '\n'
        result = result + 'VAE Diagnosis:            ' + str(self.vae_results) + '\n'
        result = result + 'Associativity Diagnosis:  ' + str(self.assoc_results) + '\n'
        result = result + 'POMDP Diagnosis:          ' + str(self.pomdp_results) + '\n'
        return result

