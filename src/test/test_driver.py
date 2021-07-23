""" Test Driver """
import os
import sys
import unittest

from driver import * 

class TestDriver(unittest.TestCase):

    def setUp(self):
        self.test_path = os.path.dirname(os.path.abspath(__file__)) + '/../../'

    def test_main(self):
        # os.system('python3 ' + self.test_path + 'driver.py -t' )
        return 

    def test_run_unit_tests(self):
        return 

    def test_init_global_paths(self):
        return 

if __name__ == '__main__':
    unittest.main()
