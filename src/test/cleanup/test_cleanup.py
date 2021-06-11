""" Test Cleanup Functionality """
import os
import unittest

import src.util.cleanup as cleanup

class TestCleanup(unittest.TestCase):

    def setUp(self):
        pass

    def test_pre_run_cleanup(self):
        file = open(".DS_Store", "w")
        file.write("hello world")
        file.flush()
        file.close()
        self.assertTrue(os.path.isfile(".DS_Store")) # Junk file to clean exists

        cleanup.clean(True) # PreRun Cleanup
        self.assertTrue(not os.path.isfile(".DS_Store")) # Junk file to clean no longer exists
        self.assertTrue(os.path.isdir("results")) # Results dir has been made

        cleanup.clean(True, path='src/test/') # PreRun Cleanup (using path variable)
        self.assertTrue(os.path.isdir("src/test/results")) # Results dir has been made (using path variable)

    def test_post_run_cleanup(self):
        cleanup.clean(False) # PostRun Cleanup
        self.assertTrue(True) # PostRun Cleanup doesn't do anything right now

if __name__ == '__main__':
    unittest.main()
