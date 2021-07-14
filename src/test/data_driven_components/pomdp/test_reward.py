""" Test Reward Functionality """
import os
import unittest
import random

import src.data_driven_components.pomdp.reward as reward

class TestReward(unittest.TestCase):

    def setUp(self):
        pass

    def test_reward(self):
        config = ([['A', 'B', 'C', 'D', 'E'], ['ignore', 'data', 'data', 'data', 'label'], ['', '3', '10', '-50', ''], ['', '5', '20', '50', '']], {'A': ['ignore', '', ''], 'B': ['data', '3', '5'], 'C': ['data', '10', '20'], 'D': ['data', '-50', '50'], 'E': ['label', '', '']})
        rewards = [random.randint(5, 10), random.randint(-10, 10), random.randint(-3, -1)]
        no_error_frame = [0, 3, 10, 10, 0]
        error_frame = [0, 10, 10, 10, 1]

        action = "view_C"
        data = [error_frame, error_frame, error_frame, error_frame, error_frame]
        reward_ret, answer_ret, reward.get_reward(action, data, rewards, config)
        self.assertEquals(reward_ret, rewards[2])
        self.assertEquals(answer_ret, -1)

        action = "report_error"
        data = [error_frame, error_frame, error_frame, error_frame, error_frame]
        reward_ret, answer_ret, reward.get_reward(action, data, rewards, config)
        self.assertEquals(reward_ret, rewards[0])
        self.assertEquals(answer_ret, 1)

        action = "report_no_error"
        data = [error_frame, error_frame, error_frame, error_frame, error_frame]
        reward_ret, answer_ret, reward.get_reward(action, data, rewards, config)
        self.assertEquals(reward_ret, rewards[1])
        self.assertEquals(answer_ret, 0)

        action = "view_B"
        data = [no_error_frame, no_error_frame, no_error_frame, no_error_frame, no_error_frame]
        reward_ret, answer_ret, reward.get_reward(action, data, rewards, config)
        self.assertEquals(reward_ret, rewards[2])
        self.assertEquals(answer_ret, -1)

        action = "report_error"
        data = [no_error_frame, no_error_frame, no_error_frame, no_error_frame, no_error_frame]
        reward_ret, answer_ret, reward.get_reward(action, data, rewards, config)
        self.assertEquals(reward_ret, rewards[1])
        self.assertEquals(answer_ret, 1)

        action = "report_no_error"
        data = [no_error_frame, no_error_frame, no_error_frame, no_error_frame, no_error_frame]
        reward_ret, answer_ret, reward.get_reward(action, data, rewards, config)
        self.assertEquals(reward_ret, rewards[0])
        self.assertEquals(answer_ret, 0)

if __name__ == '__main__':
    unittest.main()
