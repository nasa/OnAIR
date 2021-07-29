from numpy.testing._private.utils import assert_array_equal
from src.data_driven_components.pomdp.ppo_model import PPOModel
import unittest
import numpy as np

class TestPPOModel(unittest.TestCase):

    def setUp(self):
        self.window_size = 2
        self.ppo = PPOModel([], self.window_size, print_on=False)

    def test_apriori_training(self):
        attributes = len(self.ppo.agent.config.keys())
        self.ppo.apriori_training(np.random.rand(1,self.window_size,attributes))

    def update(self):
        attributes = len(self.ppo.agent.config.keys())
        self.ppo.update(np.random.rand(1,self.window_size,attributes))

if __name__ == '__main__':
    unittest.main()
