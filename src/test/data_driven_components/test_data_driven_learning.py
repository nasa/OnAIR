""" Test DataDrivenLearning Functionality """
import os
import unittest

from src.data_driven_components.data_driven_learning import DataDrivenLearning

class TestDataDrivenLearning(unittest.TestCase):

    def setUp(self):
        self.test_path = os.path.dirname(os.path.abspath(__file__))

    def test_init_empty_ddl(self):
        L = DataDrivenLearning()
        self.assertEqual(L.headers, [])

    def test_init_nonempty_ddl(self):
        L = DataDrivenLearning(['A'])
        self.assertEqual(L.headers, ['A'])
        self.assertEqual(L.init_learning_systems(['A'])[0], [0.0])
        self.assertEqual(L.init_learning_systems(['A'])[1], [0.0, 0.0, 0.0, 1.0])

        L2 = DataDrivenLearning(['A'], [1])
        self.assertEqual(L2.headers, ['A'])
        self.assertEqual(L2.init_learning_systems(['A'], [1])[0], [1.0])
        self.assertEqual(L2.init_learning_systems(['A'], [1])[1], [0.0, 0.0, 0.0, 1.0])

    def test_update(self):
        headers = ['TIME', 'A', 'B']
        sample = [1, 2, 3]

        L = DataDrivenLearning(headers, sample)
        _in, _out = L.update([5,6,7], '---')

        self.assertEqual(_in, [5.0, 6.0, 7.0])
        self.assertEqual(_out, [0.0, 0.0, 0.0, 1.0])

    def test_apriori_training(self):
        return


if __name__ == '__main__':
    unittest.main()
