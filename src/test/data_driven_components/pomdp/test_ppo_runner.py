""" Test PPO_Runner Functionality """
import os
import unittest
import random
import numpy as np

from src.data_driven_components.pomdp.ppo import PPO
from src.data_driven_components.pomdp.ppo_runner import PPO_Runner

class TestPPO_Runner(unittest.TestCase):

    def setUp(self):
        self.save_path = os.path.dirname(__file__) + "/models"
        self.data_path = os.path.dirname(__file__) + "/../../data/raw_telemetry_data/test_data/"
        if self.save_path[0] == "/":
            self.save_path = path[1:]
        self.ppo = PPO("test", self.save_path, config_path=self.data_path + "config.csv", print_on=False, save_me=True, reportable_states=['no_error', 'error'], alpha=0.01, discount=0.8, epsilon=0.2, run_limit=-1, reward_correct=100, reward_incorrect=-100, reward_action=-1)
        self.ppo_runner = PPO_Runner("test", self.save_path, config_path=self.data_path + "config.csv")
        self.true_data = [['A', 'B', 'C', 'D', 'E'], ['0', '3', '10', '-50', '0'], ['1', '3', '10', '-50', '0'], ['2', '4', '15', '0', '0'], ['3', '4', '15', '0', '0'], ['4', '5', '20', '50', '0'], ['5', '5', '20', '50', '0']]
        self.true_mass_data = [{'A': ['0', '1', '2'], 'B': ['3', '3', '4'], 'C': ['10', '10', '15'], 'D': ['-50', '-50', '0'], 'E': ['0', '0', '0']}, {'A': ['1', '2', '3'], 'B': ['3', '4', '4'], 'C': ['10', '15', '15'], 'D': ['-50', '0', '0'], 'E': ['0', '0', '0']}, {'A': ['2', '3', '4'], 'B': ['4', '4', '5'], 'C': ['15', '15', '20'], 'D': ['0', '0', '50'], 'E': ['0', '0', '0']}, {'A': ['0', '1', '2'], 'B': ['3', '7', '8'], 'C': ['10', '30', '40'], 'D': ['-50', '100', '200'], 'E': ['0', '1', '1']}, {'A': ['1', '2', '3'], 'B': ['7', '8', '8'], 'C': ['30', '40', '40'], 'D': ['100', '200', '200'], 'E': ['1', '1', '1']}, {'A': ['2', '3', '4'], 'B': ['8', '8', '7'], 'C': ['40', '40', '30'], 'D': ['200', '200', '100'], 'E': ['1', '1', '1']}]

    def test_current_state(self):
        self.assertEquals(self.ppo_runner.states[self.ppo_runner.current_state_index], self.ppo_runner.get_current_state())

    def test_running(self):
        self.ppo_runner.diagnose_frames(self.true_data)
        self.assertTrue(True)

    def test_cohens_kappa(self):
        self.ppo_runner.calculate_confusion_matrix(self.true_mass_data)
        self.ppo_runner.calculate_kappa(self.true_mass_data)
        self.ppo_runner.get_confusion_matrix()
        self.ppo_runner.get_kappa()
        self.assertTrue(True)

    def test_set_print_on(self):
        self.ppo_runner.set_print_on(True)
        self.assertTrue(self.ppo_runner.print_on)
        self.ppo_runner.set_print_on(False)
        self.assertTrue(not self.ppo_runner.print_on)

    def test_relu(self):
        result = self.ppo_runner.relu([1, 2, 3, -1, -2, -3])
        self.assertEquals(result, [1, 2, 3, 0, 0, 0])
        result = self.ppo_runner.relu_helper(5)
        self.assertEquals(result, 5)
        result = self.ppo_runner.relu_helper(-5)
        self.assertEquals(result, 0)

    def test_softmax(self):
        answer = np.array([0.01165623, 0.03168492, 0.08612854, 0.23412166, 0.63640865])
        result = np.array(self.ppo_runner.softmax([1, 2, 3, 4, 5]))
        self.assertEquals(result, answer)

    def test_categorical_sample(self):
        answer = self.ppo_runner.categorical_sample(np.array([0.01165623, 0.03168492, 0.08612854, 0.23412166, 0.63640865]))
        new_answer = False
        for i in range(500): # Statistically, it is virtually impossible for this sampling method to yield the same result 500 times in a row for these probabilities
            if answer != self.ppo_runner.categorical_sample(np.array([0.01165623, 0.03168492, 0.08612854, 0.23412166, 0.63640865])):
                new_answer = True
                break
        self.assertTrue(new_answer)

if __name__ == '__main__':
    unittest.main()
