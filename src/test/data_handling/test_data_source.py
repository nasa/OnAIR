""" Test DataSource Functionality """
import os
import unittest

from src.data_handling.data_source import DataSource

class TestDataSource(unittest.TestCase):

    def setUp(self):
        # self.test_path = os.path.dirname(os.path.abspath(__file__))
        self.empty_D = DataSource()
        self.nonempty_D = DataSource([['1'], ['2'], ['3']])

    def test_init_empty_data_source(self):
        self.assertEqual(self.empty_D.index, 0)
        self.assertEqual(self.empty_D.data, [])
        self.assertEqual(self.empty_D.data_dimension, 0)

    def test_init_non_empty_data_source(self):
        self.assertEqual(self.nonempty_D.index, 0)
        self.assertEqual(self.nonempty_D.data, [['1'], ['2'], ['3']])
        self.assertEqual(self.nonempty_D.data_dimension, 1)

    def test_get_next(self):
        next = self.nonempty_D.get_next()
        self.assertEqual(self.nonempty_D.index, 1)
        self.assertEqual(next, ['1'])

    def test_has_more(self):
        empty_answer = self.empty_D.has_more()
        non_empty_answer = self.nonempty_D.has_more()

        self.assertEqual(empty_answer, False)
        self.assertEqual(non_empty_answer, True)

    def test_has_data(self):
        empty_answer = self.empty_D.has_more()
        non_empty_answer = self.nonempty_D.has_more()
        self.assertEqual(empty_answer, False)
        self.assertEqual(non_empty_answer, True)

        D = DataSource([['TIME', '-'], ['TIME', '-'], ['TIME', '-']])
        answer = D.has_data()
        self.assertEqual(answer, False)

        D = DataSource([['TIME'], ['TIME'], ['TIME']])
        answer = D.has_data()
        self.assertEqual(answer, False)

        D = DataSource([[], [], []])
        answer = D.has_data()
        self.assertEqual(answer, False)


        D = DataSource()
        answer = D.has_data()
        self.assertEqual(answer, False)

if __name__ == '__main__':
    unittest.main()


