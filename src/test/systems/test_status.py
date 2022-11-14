""" Test Status Functionality """
import os
import unittest

from src.systems.status import Status

class TestStatus(unittest.TestCase):

    def setUp(self):
        self.test_path = os.path.dirname(os.path.abspath(__file__))
        self.S = Status()

    def test_init_empty_spacecraft(self):
        self.assertEqual(self.S.name, 'MISSION')
        self.assertEqual(self.S.status, '---')
        self.assertEqual(self.S.bayesian_conf, -1.0)

    def test_set_status(self):
        self.S.set_status('RED', 0.5)

        self.assertEqual(self.S.status, 'RED')
        self.assertEqual(self.S.bayesian_conf, 0.5)


    def test_get_status(self):
        self.assertEqual(self.S.get_status(), '---')

    def test_get_bayesian_status(self):
        self.assertEqual(self.S.get_bayesian_status(), ('---', -1.0))

    def test_get_name(self):
        self.assertEqual(self.S.get_name(), 'MISSION')


if __name__ == '__main__':
    unittest.main()
