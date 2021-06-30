""" Test DataDrivenLearning Functionality """
import os
import unittest

from src.data_driven_components.data_driven_learning import DataDrivenLearning

class TestDataDrivenLearning(unittest.TestCase):

    def setUp(self):
        self.test_path = os.path.dirname(os.path.abspath(__file__))

    def test_init(self):
        L = DataDrivenLearning(['A'])
        self.assertEquals(L.headers, ['A'])

        # self.assertEquals(L.init_learning_systems(['A'])[0], [0.0])
        # self.assertEquals(L.init_learning_systems(['A'])[1], [0.0, 0.0, 0.0, 1.0])

        L2 = DataDrivenLearning(['A'], [1])
        self.assertEquals(L2.headers, ['A'])
        
        # self.assertEquals(L2.init_learning_systems(['A'], [1])[0], [1.0])
        # self.assertEquals(L2.init_learning_systems(['A'], [1])[1], [0.0, 0.0, 0.0, 1.0])

    def test_update(self):
        headers = ['TIME', 'A', 'B']
        sample = [1, 2, 3]

        L = DataDrivenLearning(headers, sample)
        _in, _out = L.update([5,6,7], '---')

        self.assertEquals(_in, [5.0, 6.0, 7.0])
        self.assertEquals(_out, [0.0, 0.0, 0.0, 1.0])

    def test_apriori_training(self):
        return


if __name__ == '__main__':
    unittest.main()
