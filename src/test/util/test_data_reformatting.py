""" Test Data Reformatting Functionality """
import os
import unittest
import numpy as np

from src.util.data_reformatting import *

class TestDataReformatting(unittest.TestCase):

    def setUp(self):
        self.run_path = os.environ['RUN_PATH']
        self.test_path = os.path.dirname(os.path.abspath(__file__))

    def test_prep_apriori_training_data(self):
        data = [[1,1,1,1],
                [2,2,2,2],
                [3,3,3,3],
                [4,4,4,4],
                [5,5,5,5]]
        window_size = 3

        batch_data = prep_apriori_training_data(data, window_size)

        self.assertEquals(type(batch_data),list)
        self.assertEquals(type(batch_data[0]),np.ndarray)
        self.assertEquals(type(batch_data[0][0]),np.ndarray)
        
        self.assertTrue(len(batch_data[0]) == 3)

        self.assertEquals(list(batch_data[0][0]),[1.0,1.0,1.0,1.0])
        self.assertEquals(list(batch_data[0][1]),[2.0,2.0,2.0,2.0])
        self.assertEquals(list(batch_data[0][2]),[3.0,3.0,3.0,3.0])


    def test_floatify_input(self):
        inp = [1.0, 1, '1:11', 'TEMP']
        self.assertEquals(floatify_input(inp), [1.0, 1.0, 111.0, 0.0])
        self.assertEquals(floatify_input(inp, True), [1.0, 1.0, 111.0])


if __name__ == '__main__':
    unittest.main()

