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
        self.assertEquals(self.empty_D.index, 0)
        self.assertEquals(self.empty_D.data, [])
        self.assertEquals(self.empty_D.data_dimension, 0)

    def test_init_non_empty_data_source(self):
        self.assertEquals(self.nonempty_D.index, 0)
        self.assertEquals(self.nonempty_D.data, [['1'], ['2'], ['3']])
        self.assertEquals(self.nonempty_D.data_dimension, 1)

    def test_get_next(self):
        next = self.nonempty_D.get_next()
        self.assertEquals(self.nonempty_D.index, 1)
        self.assertEquals(next, ['1'])

    def test_has_more(self):
        empty_answer = self.empty_D.has_more()
        non_empty_answer = self.nonempty_D.has_more()

        self.assertEquals(empty_answer, False)
        self.assertEquals(non_empty_answer, True)

    def test_has_data(self):
        empty_answer = self.empty_D.has_more()
        non_empty_answer = self.nonempty_D.has_more()
        self.assertEquals(empty_answer, False)
        self.assertEquals(non_empty_answer, True)

        D = DataSource([['TIME', '-'], ['TIME', '-'], ['TIME', '-']])
        answer = D.has_data()
        self.assertEquals(answer, False)

        D = DataSource([['TIME'], ['TIME'], ['TIME']])
        answer = D.has_data()
        self.assertEquals(answer, False)

        D = DataSource([[], [], []])
        answer = D.has_data()
        self.assertEquals(answer, False)


        D = DataSource()
        answer = D.has_data()
        self.assertEquals(answer, False)

if __name__ == '__main__':
    unittest.main()


