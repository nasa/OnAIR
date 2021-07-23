""" Test POMDP Util Functionality """
import os
import unittest
import random

import src.data_driven_components.pomdp.pomdp_util as util

class TestPOMDPUtil(unittest.TestCase):

    def setUp(self):
        self.data_path = os.path.dirname(__file__) + "/../../data/raw_telemetry_data/test_data/"
        self.true_data = [['A', 'B', 'C', 'D', 'E'], ['0', '3', '10', '-50', '0'], ['1', '3', '10', '-50', '0'], ['2', '4', '15', '0', '0'], ['3', '4', '15', '0', '0'], ['4', '5', '20', '50', '0'], ['5', '5', '20', '50', '0']]
        self.true_config = {'A': ['ignore', '', ''], 'B': ['data', '3', '5'], 'C': ['data', '10', '20'], 'D': ['data', '-50', '50'], 'E': ['label', '', '']}
        self.enhanced_config = {'A': ['ignore', '', ''], 'B': ['data', '3', '5', 0], 'C': ['data', '10', '20', 1], 'D': ['data', '-50', '50', 2], 'E': ['label', '', '']}
        self.true_split_data = [[['0', '3', '10', '-50', '0'], ['1', '3', '10', '-50', '0'], ['2', '4', '15', '0', '0']], [['1', '3', '10', '-50', '0'], ['2', '4', '15', '0', '0'], ['3', '4', '15', '0', '0']], [['2', '4', '15', '0', '0'], ['3', '4', '15', '0', '0'], ['4', '5', '20', '50', '0']], [['3', '4', '15', '0', '0'], ['4', '5', '20', '50', '0'], ['5', '5', '20', '50', '0']]]
        self.true_mass_data = [{'A': ['0', '1', '2'], 'B': ['3', '3', '4'], 'C': ['10', '10', '15'], 'D': ['-50', '-50', '0'], 'E': ['0', '0', '0']}, {'A': ['1', '2', '3'], 'B': ['3', '4', '4'], 'C': ['10', '15', '15'], 'D': ['-50', '0', '0'], 'E': ['0', '0', '0']}, {'A': ['2', '3', '4'], 'B': ['4', '4', '5'], 'C': ['15', '15', '20'], 'D': ['0', '0', '50'], 'E': ['0', '0', '0']}, {'A': ['0', '1', '2'], 'B': ['3', '7', '8'], 'C': ['10', '30', '40'], 'D': ['-50', '100', '200'], 'E': ['0', '1', '1']}, {'A': ['1', '2', '3'], 'B': ['7', '8', '8'], 'C': ['30', '40', '40'], 'D': ['100', '200', '200'], 'E': ['1', '1', '1']}, {'A': ['2', '3', '4'], 'B': ['8', '8', '7'], 'C': ['40', '40', '30'], 'D': ['200', '200', '100'], 'E': ['1', '1', '1']}]

    def test_mass_load_data(self):
        dict_config, data = util.mass_load_data(self.data_path, 3)
        self.assertEquals(dict_config, self.true_config)
        self.assertEquals(data, self.true_mass_data)

    def test_load_save_data(self):
        data = util.load_data(self.data_path + "mission_1.csv")
        self.assetEquals(self.true_data, data)

        util.save_data(self.data_path + "mission_3.csv", data)
        data = util.load_data(self.data_path + "mission_3.csv")
        self.assetEquals(self.true_data, data)

        os.system("rm " + self.data_path + "mission_3.csv")

    def test_load_config(self):
        config = util.load_config(self.data_path + "config.csv")
        self.assertEquals(config, self.true_config)

    def test_split_by_lookback(self):
        split_data = util.split_by_lookback(self.true_data)
        self.assertEquals(split_data, self.true_split_data)

    def test_dict_sort_data(self):
        sorted_data = util.dict_sort_data(self.enhanced_config, self.true_split_data)
        true_sorted_data = [{'A': ['0', '1', '2'], 'B': ['3', '3', '4'], 'C': ['10', '10', '15'], 'D': ['-50', '-50', '0'], 'E': ['0', '0', '0']}, {'A': ['1', '2', '3'], 'B': ['3', '4', '4'], 'C': ['10', '15', '15'], 'D': ['-50', '0', '0'], 'E': ['0', '0', '0']}, {'A': ['2', '3', '4'], 'B': ['4', '4', '5'], 'C': ['15', '15', '20'], 'D': ['0', '0', '50'], 'E': ['0', '0', '0']}, {'A': ['3', '4', '5'], 'B': ['4', '5', '5'], 'C': ['15', '20', '20'], 'D': ['0', '50', '50'], 'E': ['0', '0', '0']}]
        self.assertEquals(sorted_data, true_sorted_data)

    def test_stratified_sampling(self):
        true_samples = [{'A': ['1', '2', '3'], 'B': ['7', '8', '8'], 'C': ['30', '40', '40'], 'D': ['100', '200', '200'], 'E': ['1', '1', '1']}, {'A': ['1', '2', '3'], 'B': ['3', '4', '4'], 'C': ['10', '15', '15'], 'D': ['-50', '0', '0'], 'E': ['0', '0', '0']}, {'A': ['2', '3', '4'], 'B': ['8', '8', '7'], 'C': ['40', '40', '30'], 'D': ['200', '200', '100'], 'E': ['1', '1', '1']}, {'A': ['2', '3', '4'], 'B': ['4', '4', '5'], 'C': ['15', '15', '20'], 'D': ['0', '0', '50'], 'E': ['0', '0', '0']}]
        sorted_data = [{'A': ['0', '1', '2'], 'B': ['3', '3', '4'], 'C': ['10', '10', '15'], 'D': ['-50', '-50', '0'], 'E': ['0', '0', '0']}, {'A': ['1', '2', '3'], 'B': ['3', '4', '4'], 'C': ['10', '15', '15'], 'D': ['-50', '0', '0'], 'E': ['0', '0', '0']}, {'A': ['2', '3', '4'], 'B': ['4', '4', '5'], 'C': ['15', '15', '20'], 'D': ['0', '0', '50'], 'E': ['0', '0', '0']}, {'A': ['0', '1', '2'], 'B': ['3', '7', '8'], 'C': ['10', '30', '40'], 'D': ['-50', '100', '200'], 'E': ['0', '1', '1']}, {'A': ['1', '2', '3'], 'B': ['7', '8', '8'], 'C': ['30', '40', '40'], 'D': ['100', '200', '200'], 'E': ['1', '1', '1']}, {'A': ['2', '3', '4'], 'B': ['8', '8', '7'], 'C': ['40', '40', '30'], 'D': ['200', '200', '100'], 'E': ['1', '1', '1']}]
        samples = util.stratified_sampling(self.enhanced_config, sorted_data, print_on=False)
        self.assertEquals(true_samples, samples)

    def test_check_label(self):
        label = util.check_label(self.enhanced_config)
        self.assertEquals('E', label)

    def test_split_headers(self):
        headers = ['a', 'b', 'c', 'd', 'e']
        twos = [['a', 'b'], ['a', 'c'], ['a', 'd'], ['a', 'e'], ['b', 'c'], ['b', 'd'], ['b', 'e'], ['c', 'd'], ['c', 'e'], ['d', 'e']]
        threes = [['a', 'b', 'c'], ['a', 'b', 'd'], ['a', 'b', 'e'], ['a', 'c', 'd'], ['a', 'c', 'e'], ['a', 'd', 'e'], ['b', 'c', 'd'], ['b', 'c', 'e'], ['b', 'd', 'e'], ['c', 'd', 'e']]
        fours = [['a', 'b', 'c', 'd'], ['a', 'b', 'c', 'e'], ['a', 'b', 'd', 'e'], ['a', 'c', 'd', 'e'], ['b', 'c', 'd', 'e']]

        self.assertEquals(twos, util.split_headers(headers, 2))
        self.assertEquals(threes, util.split_headers(headers, 3))
        self.assertEquals(fours, util.split_headers(headers, 4))

if __name__ == '__main__':
    unittest.main()
