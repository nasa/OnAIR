""" Test PPO Functionality """
import os
import unittest
import random

from src.data_driven_components.pomdp.ppo import PPO

class TestPPO(unittest.TestCase):

    def setUp(self):
        self.save_path = os.path.dirname(__file__) + "/models"
        self.data_path = os.path.dirname(__file__) + "/../../data/raw_telemetry_data/test_data/"
        if self.save_path[0] == "/":
            self.save_path = path[1:]
        self.ppo = PPO("test", self.save_path, config_path=self.data_path + "config.csv", print_on=False, save_me=True, reportable_states=['no_error', 'error'], alpha=0.01, discount=0.8, epsilon=0.2, run_limit=-1, reward_correct=100, reward_incorrect=-100, reward_action=-1)
        self.true_data = [['A', 'B', 'C', 'D', 'E'], ['0', '3', '10', '-50', '0'], ['1', '3', '10', '-50', '0'], ['2', '4', '15', '0', '0'], ['3', '4', '15', '0', '0'], ['4', '5', '20', '50', '0'], ['5', '5', '20', '50', '0']]
        self.true_mass_data = [{'A': ['0', '1', '2'], 'B': ['3', '3', '4'], 'C': ['10', '10', '15'], 'D': ['-50', '-50', '0'], 'E': ['0', '0', '0']}, {'A': ['1', '2', '3'], 'B': ['3', '4', '4'], 'C': ['10', '15', '15'], 'D': ['-50', '0', '0'], 'E': ['0', '0', '0']}, {'A': ['2', '3', '4'], 'B': ['4', '4', '5'], 'C': ['15', '15', '20'], 'D': ['0', '0', '50'], 'E': ['0', '0', '0']}, {'A': ['0', '1', '2'], 'B': ['3', '7', '8'], 'C': ['10', '30', '40'], 'D': ['-50', '100', '200'], 'E': ['0', '1', '1']}, {'A': ['1', '2', '3'], 'B': ['7', '8', '8'], 'C': ['30', '40', '40'], 'D': ['100', '200', '200'], 'E': ['1', '1', '1']}, {'A': ['2', '3', '4'], 'B': ['8', '8', '7'], 'C': ['40', '40', '30'], 'D': ['200', '200', '100'], 'E': ['1', '1', '1']}]

    def test_load_save_model(self):
        file_paths = [self.save_path + "pomdp_model_test.pkl", self.save_path + "pomdp_model_test_actor_policy_state_dict.pt", self.save_path + "pomdp_model_test_critic_policy_state_dict.pt", self.save_path + "ppo_weights_test.pkl"]
        for e in file_paths:
            os.system("rm " + e)
        for e in file_paths:
            self.assertTrue(not os.path.isfile(e))
        self.ppo = PPO("test", self.save_path, config_path=self.data_path + "config.csv", print_on=False, save_me=True, reportable_states=['no_error', 'error'], alpha=0.01, discount=0.8, epsilon=0.2, run_limit=-1, reward_correct=100, reward_incorrect=-100, reward_action=-1)
        self.ppo.save_PPO()
        for e in file_paths:
            self.assertTrue(os.path.isfile(e))
        data = self.ppo.get_save_data()
        self.ppo = POMDP("test", self.save_path, config_path=self.data_path + "config.csv")
        self.assertEquals(self.ppo.get_save_data(), data)

    def test_current_state(self):
        self.assertEquals(self.ppo.states[self.ppo.current_state_index], self.ppo.get_current_state())

    def test_training(self):
        self.ppo.new_training(self.true_data)
        self.assertTrue(True)

    def test_running(self):
        self.ppo.diagnose_frames(self.true_data)
        self.assertTrue(True)

    def test_cohens_kappa(self):
        self.ppo.calculate_confusion_matrix(self.true_mass_data)
        self.ppo.calculate_kappa(self.true_mass_data)
        self.ppo.get_confusion_matrix()
        self.ppo.get_kappa()
        self.assertTrue(True)

    def test_apriori_training(self):
        self.ppo.train_ppo(self.true_mass_data, 2) # This also tests plot_graph and test
        self.assertTrue(os.path.isfile(self.save_path + "pomdp_model_test.pkl"))
        self.assertTrue(os.path.isfile(self.save_path + "graph_test_Avg. Rewards.png"))
        self.assertTrue(os.path.isfile(self.save_path + "graph_test_Avg. Accuracy.png"))

    def test_set_print_on(self):
        self.ppo.set_print_on(True)
        self.assertTrue(self.ppo.print_on)
        self.ppo.set_print_on(False)
        self.assertTrue(not self.ppo.print_on)

if __name__ == '__main__':
    unittest.main()
