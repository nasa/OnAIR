""" Test Observation Functionality """
import os
import unittest


from src.data_driven_components.pomdp import observation

class TestObservation(unittest.TestCase):
    def setUp(self):
        self.test_path = os.path.dirname(os.path.abspath(__file__))

    def test_get_starting_state(self):
        config = {'VOL' : ['data',0, 5, 0], 'CUR': ['data',0, 5, 1], 'TMP': ['data',60, 100, 2]}
        starting_state = [['?', '?'],['?', '?'],['?', '?']]
        self.assertEquals(observation.get_starting_state(config), starting_state)

    def test_floatify_state(self):
        state = [['?', 'STABLE', 'BROKEN']]
        floatified_state = [1,0,0,0,1,0,0,0,1]        
        self.assertEquals(observation.floatify_state(state), floatified_state)

    def test_update_by_threshold(self):
        state = [['?', '?'],['?', '?'],['?', '?']]
        action = "view_VOL"
        data = {'VOL' : [0, 1, 2], 'CUR': [0,1,2], 'TMP': [86, 98, 21]}
        config = {'VOL' : ['data',0, 5, 0], 'CUR': ['data',0, 5, 1], 'TMP': ['data',60, 100, 2]}
        returned_state = [['VOL_THRESH_STABLE', '?'],['?', '?'],['?', '?']]
        self.assertEquals(observation.update_by_threshold(state, action, data, config), returned_state)

    def test_update_by_kalman(self):
        state = [['?', '?'],['?', '?'],['?', '?']]
        action = "view_VOL"
        data = {'VOL' : [0, 1, 2], 'CUR': [0,1,2], 'TMP': [86, 98, 21]}
        config = {'VOL' : ['data',0, 5, 0], 'CUR': ['data',0, 5, 1], 'TMP': ['data',60, 100, 2]}
        returned_state = [['?', 'VOL_KAL_STABLE'],['?', '?'],['?', '?']]
        self.assertEquals(observation.update_by_kalman(state, action, data, config), returned_state)

    def test_update_by_kalman(self):
        state = [['?', '?'],['?', '?'],['?', '?']]
        action = "view_VOL"
        data = {'VOL' : [0, 1, 2], 'CUR': [0,1,2], 'TMP': [86, 98, 21]}
        config = {'VOL' : ['data',0, 5, 0], 'CUR': ['data',0, 5, 1], 'TMP': ['data',60, 100, 2]}
        returned_state = [['VOL_THRESH_STABLE', 'VOL_KAL_STABLE'],['?', '?'],['?', '?']]
        self.assertEquals(observation.get_observation(state, action, data, config), returned_state)

    def test_strip_view_prefix(self):
        action = "view_VOL"
        stripped_action = "VOL"
        self.assertEquals(observation.strip_view_prefix(action), stripped_action)

    def test_get_attribute_threshold(self):
        attribute = "VOL"
        config = {'VOL' : ['data',0, 5, 0], 'CUR': ['data',0, 5, 1], 'TMP': ['data',60, 100, 2]}
        threhold_values = (0,5)
        self.assertEquals(observation.get_attribute_threshold(attribute, config), threhold_values)

    def test_update_state(self):
        state = [['?', '?'],['?', '?'],['?', '?']]
        attribute = "VOL" 
        observation_tool_used = "THRESH"
        answer = "STABLE"
        config = {'VOL' : ['data',0, 5, 0], 'CUR': ['data',0, 5, 1], 'TMP': ['data',60, 100, 2]}
        returned_state = [['VOL_THRESH_STABLE', '?'],['?', '?'],['?', '?']]
        self.assertEquals(observation.update_state(state, attribute, observation_tool_used, answer, config), returned_state)

    def test_get_index(self):
        attribute = "VOL" 
        config = {'VOL': ['data',0, 5, 0], 'CUR': ['data',0, 5, 1], 'TMP': ['data',60, 100, 2]}
        index = 0
        self.assertEquals(observation.get_index(attribute, config), index)

    def test_get_possible_branches(self):
        state = [['?', '?'],['?', '?'],['?', '?']]
        action = "view_VOL"
        config = {'VOL' : ['data',0, 5, 0], 'CUR': ['data',0, 5, 1], 'TMP': ['data',60, 100, 2]}
        all_possible_branches = [
            [['VOL_THRESH_STABLE', 'VOL_KAL_STABLE'],['?', '?'],['?', '?']],
            [['VOL_THRESH_STABLE', 'VOL_KAL_BROKEN'],['?', '?'],['?', '?']],
            [['VOL_THRESH_BROKEN', 'VOL_KAL_STABLE'],['?', '?'],['?', '?']],
            [['VOL_THRESH_BROKEN', 'VOL_KAL_BROKEN'],['?', '?'],['?', '?']]
        ]
        self.assertEquals(observation.get_possible_branches(state, action, config), all_possible_branches)

if __name__ == '__main__':
    unittest.main()