""" Test Spacecraft Functionality """
import os
import unittest

from src.systems.spacecraft import Spacecraft
from src.systems.status import Status
from src.data_handling.time_synchronizer import TimeSynchronizer

class TestSpacecraft(unittest.TestCase):

    def setUp(self):
        self.test_path = os.path.dirname(os.path.abspath(__file__))

        # self.SC = Spacecraft()
        # self.SC.headers = ['TIME', 'A', 'B']
        # self.SC.tests = [[['SYNC', 'TIME']], [['NOOP']], [['NOOP']]]
        # self.SC.curr_data = ['-', '-', '-']

        self.SC = Spacecraft(['TIME', 'A', 'B'], [[['SYNC', 'TIME']], [['NOOP']], [['NOOP']]])

    def test_init_empty_spacecraft(self):
        SC = Spacecraft()
        self.assertEquals(type(SC.status), Status)
        self.assertEquals(SC.headers, [])
        self.assertEquals(SC.tests, [])
        self.assertEquals(SC.curr_data, [])

    def test_init_nonempty_spacecraft(self):
        hdrs = ['TIME', 'A', 'B'] 
        tests = [[['SYNC', 'TIME']], [['NOOP']], [['NOOP']]]

        SC = Spacecraft(hdrs, tests)

        self.assertEquals(type(SC.status), Status)
        self.assertEquals(SC.headers, ['TIME', 'A', 'B'])
        self.assertEquals(SC.tests, [[['SYNC', 'TIME']], [['NOOP']], [['NOOP']]])
        self.assertEquals(SC.curr_data, ['-', '-', '-'])

    def test_update(self):
        frame = [3, 1, 5]
        self.SC.update(frame)
        self.assertEquals(self.SC.get_current_data(), [3, 1, 5])

        frame = [4, '-', 5]
        self.SC.update(frame)
        self.assertEquals(self.SC.get_current_data(), [4, 1, 5])

    def test_get_current_time(self):
        self.assertEquals(self.SC.get_current_time(), '-')

    def tests_get_status(self):
        self.assertEquals(self.SC.get_status(), '---')

    def tests_get_bayesian_status(self):
        self.assertEquals(self.SC.get_bayesian_status(), ('---', -1.0))



if __name__ == '__main__':
    unittest.main()
