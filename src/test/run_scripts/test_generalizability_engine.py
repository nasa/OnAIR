""" Test Generalizability Engine """ 

import os 
import unittest 

from src.run_scripts.generalizability_engine import GeneralizabilityEngine, DataWrapper

from src.data_driven_components.associativity import Associativity
from src.data_driven_components.vae import VAE
from src.data_driven_components.pomdp import POMDP

class TestGeneralizabilityEngine(unittest.TestCase):

    def setUp(self):
        self.test_path = os.path.dirname(os.path.abspath(__file__))
        self.run_path = os.environ['RUN_PATH']

    def test_init_empty_engine(self):
        GE = GeneralizabilityEngine()
        self.assertEquals(GE.sample_paths, [])
        self.assertEquals(GE.data_samples, [])
        self.assertEquals(GE.construct, None)

    def test_init_nonempty_engine(self):
        GE = GeneralizabilityEngine(self.run_path, 'Associativity', sample_paths=['generalizability_testing/'])

        self.assertEquals(GE.sample_paths, ['generalizability_testing/'])
        self.assertEquals(type(GE.data_samples[0]), DataWrapper)
        self.assertTrue(len(GE.data_samples) > 0)
        self.assertEquals(type(GE.construct), Associativity)

        GE = GeneralizabilityEngine(self.run_path, 'VAE', sample_paths=['generalizability_testing/'])

        self.assertEquals(GE.sample_paths, ['generalizability_testing/'])
        self.assertEquals(type(GE.data_samples[0]), DataWrapper)
        self.assertTrue(len(GE.data_samples) > 0)
        self.assertEquals(type(GE.construct), VAE)

        GE = GeneralizabilityEngine(self.run_path, 'POMDP', sample_paths=['generalizability_testing/'])

        self.assertEquals(GE.sample_paths, ['generalizability_testing/'])
        self.assertEquals(type(GE.data_samples[0]), DataWrapper)
        self.assertTrue(len(GE.data_samples) > 0)
        self.assertEquals(type(GE.construct), POMDP)

    def test_init_construct(self):
        GE = GeneralizabilityEngine()
        data_sample = DataWrapper('/path.csv', ['A', 'B'], [1,3])
        construct = GE.init_construct('Associativity', data_sample)

        # self.assertEquals(type(construct), Associativity)

if __name__ == '__main__':
    unittest.main()










