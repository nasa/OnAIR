""" Test Generic Component Core Functionality """

import unittest

from src.data_driven_components.generic_component.core import AIPlugIn

class TestCore(unittest.TestCase):

    def setUp(self):
        construct = AIPlugIn('test_component', ['test_A','test_B','test_C'])

    def test_init_empty_headers(self):
        self.assertRaises(AssertionError, AIPlugIn, 'test', [])

    def test_init_non_empty_headers(self):
        plugin = AIPlugIn('test_component', ['A'])
        self.assertEquals(plugin.headers, ['A'])

        plugin = AIPlugIn('test_component', ['A', 'B'])
        self.assertEquals(plugin.headers, ['A', 'B'])

    def apriori_training_empty_batch(self):
        construct.apriori_training([])

    def apriori_training_non_empty_batch(self):
        construct.apriori_training([[1.0,2.0]])

    def update_empty_frame(self):
        self.assertRaises(AssertionError, construct.update([]))
        
    def update_non_empty_frame(self):
        construct.update([1.0,2.0,3.0])
        # do we not do an assert here?

    def render_diagnosis(self):
        diagnosis = construct.render_diagnosis()
        self.assertIsInstance(diagnosis, list)
        if len(diagnosis>0):
            for tlm in diagnosis:
                self.assertIn(tlm, construct.headers)
        else:
            self.assertEquals(diagnosis, [])

if __name__ == '__main__':
    unittest.main()
