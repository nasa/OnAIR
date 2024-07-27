# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

import copy
import numpy as np
import random


class Diagnosis:
    """Diagnosis Class used to store and summarize diagnosis results from individaul AIComponent"""

    NO_DIAGNOSIS = "NO_DIAGNOSIS"

    def __init__(
        self,
        time_step,
        learning_system_results,
        status_confidence,
        currently_faulting_mnemonics,
        ground_truth=None,
    ) -> None:

        self.time_step = time_step
        self.status_confidence = status_confidence
        self.learning_system_results = learning_system_results
        self.currently_faulting_mnemonics = currently_faulting_mnemonics
        self.ground_truth = ground_truth

        self.has_kalman = "kalman" in learning_system_results
        self.kalman_results = (
            learning_system_results["kalman"] if self.has_kalman else None
        )

    def perform_diagnosis(self):
        """Diagnose the learning system results"""

        ret = {}
        if self.has_kalman:
            # just pick a random mnemonic for testing
            mnemonic_name = random.choice(list(self.kalman_results[0]))
            top = self.walkdown(mnemonic_name)

            ret = {"top": top}

        return ret

    def walkdown(self, mnemonic_name, used_mnemonics=[]):
        """
        Go through the active AIComponents in an ordered way to decide on a diagnosis.
        There's a lot of specificity in this function until the method of combining the AIComponents is learned
        """
        if len(used_mnemonics) == 0:
            used_mnemonics = copy.deepcopy(self.currently_faulting_mnemonics)

        if mnemonic_name == "":
            return Diagnosis.NO_DIAGNOSIS

        if self.has_kalman:
            # NOTE: This is certainly wrong since the logic is pulled from a statement with many AIComponents
            if not (mnemonic_name in list(self.kalman_results[0])):
                return self.kalman_results[0][0]
            else:
                return Diagnosis.NO_DIAGNOSIS
        else:
            return Diagnosis.NO_DIAGNOSIS
