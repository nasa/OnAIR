""" Test Driver """
import os
import unittest

class TestDriver(unittest.TestCase):

    def setUp(self):
        self.test_path = os.path.dirname(os.path.abspath(__file__)) + '/../../'

    def test_driver(self):
        # os.system('python3 ' + self.test_path + 'driver.py -t' )
        return 

if __name__ == '__main__':
    unittest.main()
