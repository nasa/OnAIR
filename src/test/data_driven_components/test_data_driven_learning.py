""" Test DataDrivenLearning Functionality """
import os
import unittest

from src.data_driven_components.data_driven_learning import DataDrivenLearning

class TestDataDrivenLearning(unittest.TestCase):

    def setUp(self):
        self.test_path = os.path.dirname(os.path.abspath(__file__))

    def test_init(self):
        L = DataDrivenLearning(['A'])
        self.assertEquals(L.headers, ['A'])

    def test_update(self):
        headers = ['TIME', 'A', 'B']

        L = DataDrivenLearning(headers)
        _in, _out = L.update([5,6,7], '---')

        self.assertEquals(_in, [5.0, 6.0, 7.0])
        self.assertEquals(_out, [0.0, 0.0, 0.0, 1.0])

    def test_apriori_training(self):
        return

    def test_update(self):
        return 
    def test_diagnose(self):
        return 

    def test_status_to_oneHot(self):
        return 

if __name__ == '__main__':
    unittest.main()
