""" Test Cleanup Functionality """
import os
import unittest

import src.util.cleanup as cleanup

class TestCleanup(unittest.TestCase):

    def setUp(self):
        pass

    def test_pre_run_cleanup(self):
        basePath = os.path.dirname(__file__) + "/../../../"
        file = open(basePath + ".DS_Store", "w")
        file.write("hello world")
        file.flush()
        file.close()
        self.assertTrue(os.path.isfile(basePath + ".DS_Store")) # Junk file to clean exists

        cleanup.clean(True) # PreRun Cleanup
        self.assertTrue(not os.path.isfile(basePath + ".DS_Store")) # Junk file to clean no longer exists
        self.assertTrue(os.path.isdir(basePath + "results")) # Results dir has been made

        cleanup.clean(True, path='src/test/') # PreRun Cleanup (using path variable)
        self.assertTrue(os.path.isdir(basePath + "src/test/results")) # Results dir has been made (using path variable)

    def test_post_run_cleanup(self):
        cleanup.clean(False) # PostRun Cleanup
        self.assertTrue(True) # PostRun Cleanup doesn't do anything right now

if __name__ == '__main__':
    unittest.main()
