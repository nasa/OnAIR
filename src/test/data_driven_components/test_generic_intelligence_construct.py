""" Test GenericIntelligenceConstruct [Abstract Class] Functionality """

import unittest

from src.data_driven_components.generic_intelligence_construct import GenericIntelligenceConstruct

class TestGenericIntelligenceConstruct(unittest.TestCase):

    def setUp(self):
        pass

    def test_init(self):
        self.assertRaises(TypeError, GenericIntelligenceConstruct, [])

    def apriori_training(self):
        pass

    def update(self):
        pass
    
    def render_diagnosis(self):
        pass

if __name__ == '__main__':
    unittest.main()
