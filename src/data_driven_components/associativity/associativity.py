'''
Associativity main class

Author: Chris Trombley
June 9th 2021
'''

import numpy as np
import matplotlib.pyplot as plt
import networkx as nwx
from efficient_apriori import apriori

from src.data_driven_components.associativity.associativity_data_manager import AssociativityDataManager

class Associativity:
    def __init__(self, headers=[], sample_input=[], prepModel=False):
        self.window_size = 20 ##make sure common vars in this and data manager are set sync
        self.frame_id = 0
        self.benchmark_file_path = None
        self.number_of_sensors = 0
        self.associations = np.zeros((self.number_of_sensors, self.number_of_sensors))
        self.activated_elements = None
        self.save_graph = False
        self.graph_nodes = []
        self.graph_weights = []
        self.graph_edges_with_weights = []
        self.graph_edges = {}
        self.associativityDataManager = AssociativityDataManager(headers, prepModel)
        
        self.verbose = False
        self.visualize_graph = False

    def update(self, frame):
        self.associativityDataManager.add_frame(frame)
        self.frame_id += 1
            # if self.visualize_graph == True:
            #     self.create_and_view_graph()

    def render_diagnosis(self):
        rules = self.compute_association_rules()
        return rules

    def compute_association_rules(self):
        data = self.associativityDataManager.get_data()
        association_rule_index = 0
        
        if data != -1:
            data = [tuple(elem[1:6]) for elem in data]

            itemsets, association_rules = apriori(data, min_support=0.9, min_confidence=0.9) ##computes the association rules
            return association_rules
        else:
            print('[associativity.py] ERROR: Records are empty. Is the window size too large?')

            # if self.verbose == True:
            #     for association in association_rules:
            #         pair = association[0]
            #         items = [x for x in pair]
            #         try:
            #             if len(items) >= 2 and len(association) >= 2:
            #                 association_rule_index += 1
            #                 print(f'\n--------------Start of Association Rule {association_rule_index}------------ ')
            #                 print('Rule: ' + items[0] + '->' + items[1])
            #                 print('Support: ' + str(association[1]))
            #                 print(f'---------------End of Association Rule {association_rule_index}----------------')
            #                 self.graph_nodes.append(items[0])
            #                 self.graph_nodes.append(items[1])
            #                 edge_weight_pairing = (items[0], items[1], {'weight': association[1]})
            #                 self.graph_edges_with_weights.append(edge_weight_pairing)
            #         except:
            #             continue ##this block is usually only called if it is a index out of bounds error
            # if self.visualize_graph == True:
            #     create_and_view_graph()

    def load_ground_truth_matrix(self):
        try:
            with open(self.benchmark_file_path) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in reader:
                    benchmark.append(row)
                processed = []
                for row in benchmark[1:]:
                    row = row[1:]
                    row = [convert_matrix_elem(elem) for elem in row]
                    processed.append(row)
                return np.asarray(processed)
        except:
            print('[associativity.py] ERROR: Could not load ground truth csv file. Does the csv file exist?')

   ##How often should we compute association rules?
    def compute_association_matrix(self, activated_elements):
        for i in range(self.number_of_sensors):
            for j in range(self.number_of_sensors):
                j = self.num_sensors - j
                if i != j:
                    if i in activated_elements and j in activated_elements:
                        self.associations[i][j] += 1

    def get_association_matrix(self):
        return self.associations

    def create_and_view_graph(self):
        associativity_graph = nwx.Graph()
        associativity_graph.add_nodes_from(self.graph_nodes)
        associativity_graph.add_edges_from(self.graph_edges_with_weights)
        edge_labelsa = nwx.get_edge_attributes(associativity_graph, 'weight')
        pos = nwx.spring_layout(associativity_graph)
        nwx.draw(associativity_graph, pos, with_labels = True)
        nwx.draw_networkx_edge_labels(associativity_graph, pos, edge_labels = edge_labelsa)

        if self.save_graph == True:
            plt.savefig('pathtofig')

        plt.show()
    
    def compute_confusion_matrix(self):
        TN, TP, FN, FP = 0, 0, 0, 0
        ground_truth = self.load_ground_truth_matrix()
        pred = self.get_association_matrix()
        for i in range(ground_truth.shape[0]):
            for j in range(ground_truth.shape[1]):
                if j > 1:
                    total_count += 1
                    if pred[i][j] > 0:
                        association_activations += 1
                    if ground_truth[i][j] > 0:
                        ground_truth_activations += 1
                    if pred[i][j] == 0 and ground_truth[i][j] == 0:
                        TN += 1
                    if pred[i][j] > 0 and ground_truth[i][j] > 0:
                        TP += 1
                    if pred[i][j] > 0 and ground_truth[i][j] == 0:
                        FP += 1
                    if pred[i][j] == 0 and ground_truth[i][j] > 0:
                        FN += 1
        return TN, TP, FN, FP