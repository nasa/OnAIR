"""
Diagnosis Class, used to store all diagnosis results / summary info 
"""

import csv
import copy 
import numpy as np

class Diagnosis:
    def __init__(self, time_step=0, 
                       learning_systems_results = {'vae_diagnosis' : None,
                                                   'associativity_diagnosis' : None,
                                                   'pomdp_diagnosis' : None,
                                                   'kalman_diagnosis' : None,
                                                   'causality_diagnosis' : None},
                       status_confidence=1.0, 
                       faulting_mnemonics=[],
                       ground_truth = None):

        self.time_step = time_step
        self.vae_results = learning_systems_results['vae_diagnosis']
        self.pomdp_results = learning_systems_results['pomdp_diagnosis']
        self.kalman_results = learning_systems_results['kalman_diagnosis']
        self.causality_results = learning_systems_results['causality_diagnosis']
        self.status_confidence = status_confidence
        self.faulting_mnemonics = faulting_mnemonics
        self.ground_truth = ground_truth 

    def perform_diagnosis(self):

        hdrs, list_of_vectors = self.generate_vectorized_values()

        abs_val_top, val_abs = self.vectorization_diagnosis(hdrs, list_of_vectors, absVal=True)
        non_abs_val_top, val_non_abs = self.vectorization_diagnosis(hdrs, list_of_vectors, absVal=False)

        walkdown_abs = self.walkdown(abs_val_top, [])
        walkdown_non_abs = self.walkdown(non_abs_val_top, [])

        diagnoses = {'abs_val_top' : abs_val_top, 
                     'non_abs_val_top' : non_abs_val_top, 
                     'walkdown_abs' : walkdown_abs,
                     'walkdown_non_abs' : walkdown_non_abs,
                     'vae_top1' : self.vae_diagnosis(1),
                     'vae_top2' : self.vae_diagnosis(2)}
                     #'pomdp_top1' : self.pomdp_diagnosis(1)}

              
        # print("Ground Truth:              " + str(gt))
        # print("Result:                    " + str(abs_val_top) + ' ' + str(val_abs))
        # print("Walkdown:                  " + str(walkdown_abs)+ '\n***************************************')
        # print("Walkdown, Non-Abs:         " + str(walkdown_non_abs))
        # print("Non Abs: " + str(non_abs_val_top) + ' ' + str(val_non_abs))
        # print(diagnoses)

        return diagnoses

    def vae_diagnosis(self, k):
        results = self.abs_vector(self.vae_results)[0]
        return list(results[:k])

    def pomdp_diagnosis(self, k):
        if((1+k) <= len(self.pomdp_results[0])):
            return [self.pomdp_results[0][-(1+k)].replace("view_", "")]
        return ['TBD']

    def generate_walkdown_traversal_tree(self, causality_results, use_abs=True):
        """
        Given causality results, generates traversal tree
        :param causality_results: {string: ([string],[int]) }
        :param use_abs: (optional bool) whether to use absolute values for causality sorting
        :returns: {string: [string]} dictionary where each mnemonic has its most 
            related values mnemonics sorted from most to least causing
        """
        traversal_tree = copy.deepcopy(causality_results)
        if use_abs:
            traversal_tree = {mnemonic: (hdr, list(map(abs, val))) for (mnemonic, (hdr, val)) in traversal_tree.items()}
        traversal_tree = {mnemonic: list(zip(*sorted(zip(*hdrval), reverse=True, key=lambda hv: hv[1])))[0] for (mnemonic, hdrval) in traversal_tree.items()}
        traversal_tree = {mnemonic: list(val) for (mnemonic, val) in traversal_tree.items()}
        return traversal_tree

    def generate_walkdown_mnemonic(self, fault_mnemonic, use_abs=False):
        """
        :param fault_mnemonic: (string) mnemonic to start walkdown from
        :param use_abs: (optional bool) whether to use absolute causality values
        :returns: (string) mnemonic after going through all walkdown steps
        """
        traversal_tree = self.generate_walkdown_traversal_tree(self.causality_results, use_abs)
        end_mnemonic = fault_mnemonic
        while not end_mnemonic == 'ALTITUDE' and not end_mnemonic in self.kalman_results[0]:
            if traversal_tree[end_mnemonic] == []:
                return end_mnemonic
            else:
                end_mnemonic = traversal_tree[end_mnemonic][0]
                for (_, val) in traversal_tree.items():
                    val.remove(end_mnemonic)
        return end_mnemonic

    def generate_walkdown_vectorization_values(self, walkdown_mnemonic):
        """
        Same as generate_vectorized_values but can specify which mnemonics values to use for causality
        :param walkdown_mnemonic: (string) mnemonic to use for causality vectorization
        """
        all_hdrs = self.vae_results[0]
        mnemonics_to_consider = [walkdown_mnemonic]

        vae = self.correspond_vector(all_hdrs, self.vae_results)[1]

        causality = [self.correspond_vector(all_hdrs, self.causality_results[x])[1] for x in mnemonics_to_consider if x in self.causality_results.keys()]

        all_vectors = copy.deepcopy(causality)
        all_vectors.append(vae)

        return all_hdrs, all_vectors


    def generate_vectorized_values(self, walkdownAndKalman=False):
        all_hdrs = self.vae_results[0]
        mnemonics_to_consider = self.faulting_mnemonics
        causality = []
        for x in mnemonics_to_consider:
            if x in self.causality_results.keys():
                causality.append(self.correspond_vector(all_hdrs, self.causality_results[x])[1])
            else:
                causality.append(self.correspond_vector(all_hdrs, ([x], [1.0]*len(all_hdrs)))[1])
        all_vectors = copy.deepcopy(causality)

        # all_vectors.append(self.vae_results[1])

        return all_hdrs, all_vectors


    def walkdown(self, mnemonic_name, used_mnemonics):
        if (used_mnemonics == []):
            used_mnemonics = copy.deepcopy(self.faulting_mnemonics)
        if len(used_mnemonics) == len(self.vae_results[0]):
            return 'NO_DIAGNOSIS'
        if mnemonic_name == '':
            return 'NO_DIAGNOSIS'
        if not mnemonic_name in self.causality_results.keys():
            if not mnemonic_name in list(self.kalman_results[0]):
                return self.kalman_results[0][0]
            else:
                return 'NO_DIAGNOSIS'
        if (mnemonic_name not in list(self.kalman_results[0])) and mnemonic_name in self.causality_results.keys():
            new_vector_tuple = copy.deepcopy(self.causality_results[mnemonic_name])
            hdrs = new_vector_tuple[0]
            vals = new_vector_tuple[1]


            for mn in used_mnemonics:
                if mn in hdrs:
                    i = hdrs.index(mn)
                    del hdrs[i]
                    del vals[i]

            if len(vals) == 0:return 'NO_DIAGNOSIS'

            max_val = max(vals)
            j = vals.index(max_val)
            name = hdrs[j]

            used_mnemonics.append(mnemonic_name)
            return self.walkdown(name, used_mnemonics)
        if not mnemonic_name in list(self.kalman_results[0]):
            return 'NO_DIAGNOSIS'        
        return mnemonic_name

    def get_most_related_mnemonic(self, mnemonic_tuple):
        abz = list([abs(x) for x in mnemonic_tuple[1]])
        top_rank = max(abz)
        i = abz.index(top_rank)
        return mnemonic_tuple[0][i]

    def vectorization_diagnosis(self, all_hdrs, all_vectors, absVal=False):
        if all_vectors == []:
            return '', 0.0

        vectorization = np.array([0.0]*len(all_hdrs))

        if absVal == True:
            all_vectors = [[abs(x) for x in elem] for elem in all_vectors]

        for res in all_vectors:
            vectorization = np.array(res) + vectorization 

        abz = list([abs(x) for x in vectorization])
        top_rank = max(abz)
        i = abz.index(top_rank)
        return all_hdrs[i], top_rank

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
                if g in self.vae_results[0][:(i+1)]:
                    acc[i] += 1
        return acc

    def abs_vector(self, vector):
        hdrs = vector[0]
        conf_vals = [abs(x) for x in vector[1]]
        ordered_vals, ordered_headers = zip(*sorted(zip(conf_vals, hdrs), reverse=True))
        return (ordered_headers, ordered_vals)

    def correspond_vector(self, hdr_order, tupled_val):
        new_vals = []
        for hdr in hdr_order:
            try:
                i = tupled_val[0].index(hdr)
                new_vals.append(tupled_val[1][i])
            except:
                new_vals.append(0.0)
        return (hdr_order, new_vals)

    def normalize(self, raw):
        norm_val = max(raw)
        if norm_val == 0.0:
            abs_val_version = [abs(x) for x in raw]
            if max(abs_val_version) == 0.0:
                return raw
            else:
                offset = [x+max(abs_val_version) for x in raw]
                raw = offset
        return [float(i)/max(raw) for i in raw]

    ##### GETTERS/SETTERS/STR ##############################
    def set_ground_truth(self, gt):
        self.ground_truth = gt

    def __str__(self):
        diagnosis = self.perform_diagnosis()
        result = ''
        result = result + '\nTime Step:                ' + str(self.time_step) + '\n'
        result = result + 'RED Status Confidence:    ' + str(self.status_confidence) + '\n'
        result = result + 'VAE Diagnosis:            ' + str(self.vae_results) + '\n'
        result = result + 'Kalman Diagnosis:         ' + str(self.kalman_results) + '\n'
        #result = result + 'POMDP Diagnosis:          ' + str(self.pomdp_results) + '\n'

        result = result + 'Causality Diagnosis:      \n' 
        for res in self.causality_results.keys():
            result = result + '                           ' + str(res) + ': ' + str(self.causality_results[res]) + '\n\n'

        result = result + 'Ground Truth:              ' + str(self.ground_truth) + '\n'
        result = result + 'Faulting Mnemonics:        ' + ',' .join(str(s) for s in self.faulting_mnemonics)
        result = result + ''
        return result

