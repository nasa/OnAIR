""" Test kalman_filter Functionality """
import os
import unittest

import simdkalman
import numpy as np

from src.data_driven_components.pomdp import kalman_filter

class TestKalmanFilter(unittest.TestCase):

    def setUp(self):
        self.test_path = os.path.dirname(os.path.abspath(__file__))

    def test_return_kf(self):
        self.assertEquals(kalman_filter.return_KF(), kalman_filter.kf)

    def test_mean(self):
        numbers = [1.0, 5.5, 6.5, 9.8, 10.12]
        mean = 6.5840000000000005
        self.assertEquals(kalman_filter.mean(numbers), mean)

    def test_absolute_value_residuals(self):
        predicted = 1.0
        actual = 1.2
        absolute_value_residuals = 0.19999999999999996
        self.assertEquals(kalman_filter.residual(predicted, actual), absolute_value_residuals)

    def test_predict_one_step_forward(self):
        numbers = [1.0, 5.5, 6.5, 9.8, 10.12]
        prediction = [13.270908135108757]
        self.assertEquals(kalman_filter.predict(kalman_filter.return_KF(), numbers, 1, numbers[0]).observations.mean, prediction)

    def test_generate_residuals_for_given_data(self):
        numbers = [1.0, 5.5, 6.5, 9.8, 10.12]
        residuals = [4.538461538461538, 3.025014212620807, 0.061911639985986255, 2.399305824910778]
        self.assertEquals(kalman_filter.generate_residuals_for_given_data(kalman_filter.return_KF(), numbers), residuals)

    def test_current_attribute_get_error(self):
        previous_data = [1.0, 5.5, 6.5, 9.8, 10.12]
        new_data = 11.2
        error = False
        residuals = [4.538461538461538, 3.025014212620807, 0.061911639985986255, 2.399305824910778, 2.0709081351087573]
        self.assertEquals(kalman_filter.current_attribute_get_error(kalman_filter.return_KF(), previous_data, new_data), (error,residuals))

    def test_current_attribute_get_error(self):
        data = [1.0, 5.5, 6.5, 9.8, 10.12, 11.2]
        result = True 
        self.assertEquals(kalman_filter.current_attribute_chunk_get_error(kalman_filter.return_KF(), data), result)


if __name__ == '__main__':
    unittest.main()