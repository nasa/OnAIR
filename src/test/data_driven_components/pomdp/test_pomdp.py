""" Test POMDP Functionality """
import os
import unittest
import random

from src.data_driven_components.pomdp.pomdp import POMDP

class TestPOMDP(unittest.TestCase):

    def setUp(self):
        self.save_path = os.path.dirname(__file__) + "/models"
        self.data_path = os.path.dirname(__file__) + "/../../data/raw_telemetry_data/test_data/"
        if self.save_path[0] == "/":
            self.save_path = path[1:]
        self.pomdp = POMDP("test", self.save_path, config_path=self.data_path + "config.csv", print_on=False, save_me=True, reportable_states=['no_error', 'error'], alpha=0.01, discount=0.8, epsilon=0.2, run_limit=-1, reward_correct=100, reward_incorrect=-100, reward_action=-1)
        self.true_data = [['A', 'B', 'C', 'D', 'E'], ['0', '3', '10', '-50', '0'], ['1', '3', '10', '-50', '0'], ['2', '4', '15', '0', '0'], ['3', '4', '15', '0', '0'], ['4', '5', '20', '50', '0'], ['5', '5', '20', '50', '0']]
        self.true_mass_data = [{'A': ['0', '1', '2'], 'B': ['3', '3', '4'], 'C': ['10', '10', '15'], 'D': ['-50', '-50', '0'], 'E': ['0', '0', '0']}, {'A': ['1', '2', '3'], 'B': ['3', '4', '4'], 'C': ['10', '15', '15'], 'D': ['-50', '0', '0'], 'E': ['0', '0', '0']}, {'A': ['2', '3', '4'], 'B': ['4', '4', '5'], 'C': ['15', '15', '20'], 'D': ['0', '0', '50'], 'E': ['0', '0', '0']}, {'A': ['0', '1', '2'], 'B': ['3', '7', '8'], 'C': ['10', '30', '40'], 'D': ['-50', '100', '200'], 'E': ['0', '1', '1']}, {'A': ['1', '2', '3'], 'B': ['7', '8', '8'], 'C': ['30', '40', '40'], 'D': ['100', '200', '200'], 'E': ['1', '1', '1']}, {'A': ['2', '3', '4'], 'B': ['8', '8', '7'], 'C': ['40', '40', '30'], 'D': ['200', '200', '100'], 'E': ['1', '1', '1']}]

    def test_load_save_model(self):
        file_path = self.save_path + "pomdp_model_test.pkl"
        os.system("rm " + file_path)
        self.assertTrue(not os.path.isfile(file_path))
        self.pomdp = POMDP("test", self.save_path, config_path=self.data_path + "config.csv", print_on=False, save_me=True, reportable_states=['no_error', 'error'], alpha=0.01, discount=0.8, epsilon=0.2, run_limit=-1, reward_correct=100, reward_incorrect=-100, reward_action=-1)
        self.pomdp.save_model()
        self.assertTrue(os.path.isfile(file_path))
        data = self.pomdp.get_save_data()
        self.pomdp = POMDP("test", self.save_path, config_path=self.data_path + "config.csv")
        self.assertEquals(self.pomdp.get_save_data(), data)

    def test_current_state(self):
        self.assertEquals(self.pomdp.states[self.pomdp.current_state_index], self.pomdp.get_current_state())

    def test_training(self):
        self.pomdp.new_training(self.true_data)
        self.assertTrue(True)

    def test_running(self):
        self.pomdp.run_test(self.true_data)
        self.assertTrue(True)

    def test_cohens_kappa(self):
        self.pomdp.calculate_confusion_matrix(self.true_mass_data)
        self.pomdp.calculate_kappa(self.true_mass_data)
        self.pomdp.get_confusion_matrix()
        self.pomdp.get_kappa()
        self.assertTrue(True)

    def test_apriori_training(self):
        self.pomdp.apriori_training(self.true_mass_data) # This also tests plot_graph and mass_test
        self.assertTrue(os.path.isfile(self.save_path + "pomdp_model_test.pkl"))
        self.assertTrue(os.path.isfile(self.save_path + "graph_test_Avg. Rewards.png"))
        self.assertTrue(os.path.isfile(self.save_path + "graph_test_Avg. Accuracy.png"))

    def test_set_print_on(self):
        self.pomdp.set_print_on(True)
        self.assertTrue(self.pomdp.print_on)
        self.pomdp.set_print_on(False)
        self.assertTrue(not self.pomdp.print_on)

    def test_show_examination_filtered(self):
        self.pomdp.set_print_on(False)
        self.pomdp.show_examination_filtered()
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
