""" Test Diagnosis Functionality """
import os
import unittest

from src.reasoning.diagnosis import Diagnosis

class TestDiagnosis(unittest.TestCase):

    def setUp(self):
        self.test_path = os.path.dirname(os.path.abspath(__file__))
        self.D = Diagnosis()

    def test_init(self):
        return

if __name__ == '__main__':
    unittest.main()
