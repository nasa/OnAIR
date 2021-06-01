import unittest
import sys 

from src.test.test_driver import TestDriver
# from src.test.run_scripts.test_execution_engine import TestExecutionEngine
from src.test.data_handling.parsers.test_forty_two_parser import TestFortyTwoParser
# from src.test.data_handling.test_binner import TestBinner
from src.test.data_handling.test_time_synchronizer import TestTimeSynchronizer

test_dict = {'driver' : 0,
             'execution_engine' : 1}

def create_suite():
    suite = []
    suite.append(unittest.TestLoader().loadTestsFromTestCase(TestDriver))
    # suite.append(unittest.TestLoader().loadTestsFromTestCase(TestExecutionEngine))
    suite.append(unittest.TestLoader().loadTestsFromTestCase(TestFortyTwoParser))
    # suite.append(unittest.TestLoader().loadTestsFromTestCase(TestBinner))
    suite.append(unittest.TestLoader().loadTestsFromTestCase(TestTimeSynchronizer))

    return suite

def run_tests(suite):
    for test in suite:
        unittest.TextTestRunner(verbosity=1).run(test)

def test_individual_suite(test):
    unittest.TextTestRunner(verbosity=1).run(test) 

if __name__ == '__main__':
    suite = create_suite()
    if len(sys.argv) == 2:
        try: 
            test_individual_suite(suite[int(sys.argv[1])])
        except: 
            print('Please enter a valid test index')
    else:
        run_tests(suite)

