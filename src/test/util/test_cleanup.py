""" Test Cleanup Functionality """
import os
import unittest

from src.util.cleanup import *

class TestCleanup(unittest.TestCase):

    def setUp(self):
        self.run_path = os.environ['RUN_PATH']
        self.test_path = os.path.dirname(os.path.abspath(__file__))

    def test_pre_run_setup(self):

        return

        # clean_all('') # PreRun Cleanup
        # self.assertTrue(not os.path.isfile(basePath + ".DS_Store")) # Junk file to clean no longer exists
        # self.assertTrue(os.path.isdir(basePath + "results")) # Results dir has been made

        # clean_all(run_path='src/test/') # PreRun Cleanup (using path variable)
        # self.assertTrue(os.path.isdir(basePath + "src/test/results")) # Results dir has been made (using path variable)

    def test_post_run_cleanup(self):
        file = open(self.test_path + ".DS_Store", "w")
        file.write("hello world")
        file.flush()
        file.close()

        self.assertTrue(os.path.isfile(self.test_path + ".DS_Store")) # Junk file to clean exists
        clean_all(self.run_path) # PostRun Cleanup
        self.assertFalse(os.path.isfile(self.test_path + ".DS_Store")) 

if __name__ == '__main__':
    unittest.main()
