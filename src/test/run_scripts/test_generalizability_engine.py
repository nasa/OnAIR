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
        self.GE = GeneralizabilityEngine(self.run_path, 'Associativity', sample_paths=['generalizability_testing/'])

    def test_init(self):
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
        construct = self.GE.init_construct('Associativity', [['A', 'B'], [1,3]])
        self.assertEquals(type(construct), Associativity)

        construct = self.GE.init_construct('VAE', [1,3])
        self.assertEquals(type(construct), VAE)

        construct = self.GE.init_construct('POMDP', ['name', '/path/name.csv', ['A', 'B']])
        self.assertEquals(type(construct), POMDP)

    def test_init_samples(self):
        self.GE.sample_paths = ['generalizability_testing/']
        data_samples = self.GE.init_samples(self.run_path + '/data/raw_telemetry_data/')

        self.assertEquals(type(data_samples[0]), DataWrapper)
        self.assertTrue(len(data_samples )> 0)

    def test_extract_dimensional_info(self):
        wrapper = DataWrapper('temp/path', ['A'], [1])
        self.GE.data_samples = [wrapper]
        dim_info = self.GE.extract_dimensional_info('Associativity')
        self.assertEquals(dim_info, [['A'], 1])

        dim_info = self.GE.extract_dimensional_info('POMDP')
        self.assertEquals(dim_info, ['path', 'temp/path', ['A'] ])

        dim_info = self.GE.extract_dimensional_info('VAE')
        self.assertEquals(dim_info, [1,1])

if __name__ == '__main__':
    unittest.main()










