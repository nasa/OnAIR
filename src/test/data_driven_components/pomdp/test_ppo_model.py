from numpy.testing._private.utils import assert_array_equal
from src.data_driven_components.pomdp.ppo_model import PPOModel
import unittest
import numpy as np

class TestPPOModel(unittest.TestCase):

    def setUp(self):
        self.ppo = PPOModel([], 2)

    def test_apriori_training(self):
        self.ppo.config = {'Time': ['ignore', '', ''], 'VOLTAGE': ['data', '13', '18', 0], 'CURRENT': ['data', '3', '5', 1], 'THRUST': ['data', '-200', '200', 2], 'ALTITUDE': ['data', '1200', '2800', 3], 'ACCELERATION': ['data', '-20', '20', 4], 'TEMPERATURE': ['data', '10', '90', 5], 'SCIENCE_COLLECTION': ['data', '0', '1', 6], '[LABEL]: ERROR_STATE': ['label', '', '']}
        self.ppo.apriori_training(np.random.rand(1,2,8))
    def update(self):
        self.ppo.update(np.random.rand(1,2,8))

if __name__ == '__main__':
    unittest.main()