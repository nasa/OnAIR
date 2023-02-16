import copy
import numpy as np
import random 

NO_DIAGNOSIS = "NO_DIAGNOSIS"

class Diagnosis:
    """ Diagnosis Class used to store and summarize diagnosis results from individaul AIComponent"""
    def __init__(self, 
                 time_step, 
                 learning_system_results, 
                 status_confidence,
                 currently_faulting_mnemonics, 
                 ground_truth=None) -> None:

        self.time_step = time_step
        self.status_confidence = status_confidence
        self.learning_system_results = learning_system_results
        self.currently_faulting_mnemonics = currently_faulting_mnemonics
        self.ground_truth = ground_truth

        self.kalman_results = learning_system_results["kalman_plugin"] if "kalman_plugin" in learning_system_results else None
        
    def perform_diagnosis(self):
        """ Diagnose the learning system results """
       
        # just pick a random mnemonic for testing
        mnemonic_name = random.choice(list(self.kalman_results[0]))
        top = self.walkdown(mnemonic_name)

        return {
            "top": top
        }

        
    def walkdown(self, mnemonic_name, used_mnemonics=[]):
        """ 
        Go through the active AIComponents in an ordered way to decide on a diagnosis. 
        There's a lot of specificity in this function until the method of combining the AIComponents is learned   
        """
        if len(used_mnemonics) == 0:
            used_mnemonics = copy.deepcopy(self.currently_faulting_mnemonics)

        if mnemonic_name == '':
            return NO_DIAGNOSIS

        if self.kalman_results is not None:
            # NOTE: This is certainly wrong since the logic is pulled from a statement with many AIComponents
            if not (mnemonic_name in list(self.kalman_results[0])):
                return self.kalman_results[0][0]
            else: return NO_DIAGNOSIS
        
        return mnemonic_name

        
