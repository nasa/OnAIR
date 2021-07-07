""" Test Cleanup Functionality """
import os
import unittest
import shutil

from src.util.cleanup import *

class TestCleanup(unittest.TestCase):

    def setUp(self):
        self.run_path = os.environ['RUN_PATH']
        self.test_path = os.path.dirname(os.path.abspath(__file__))

    def test_clean_all(self):
        file = open(self.test_path + ".DS_Store", "w")
        file.write("hello world")
        file.flush()
        file.close()

        self.assertTrue(os.path.isfile(self.test_path + ".DS_Store")) # Junk file to clean exists
        clean_all(self.run_path) # PostRun Cleanup
        self.assertFalse(os.path.isfile(self.test_path + ".DS_Store")) 

    def test_setup_folders(self):
        path = self.test_path + '/results/'
        setup_folders(path)
        self.assertTrue(os.path.isdir(path))
        shutil.rmtree(path)
        self.assertFalse(os.path.isdir(path))

if __name__ == '__main__':
    unittest.main()
