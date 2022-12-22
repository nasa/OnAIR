""" Test DataSource Functionality """
import pytest
from mock import MagicMock

import src.data_handling.data_source as data_source
from src.data_handling.data_source import DataSource

# __init__ tests
def test__init__DataSource_sets_index_to_0_and_data_to_given_data_and_data_dimension_to_len_of_first_data_in_data_when_given_data_occupied_and_first_data_has_len_more_than_0():
    # Arrange
    arg_data = []

    num_fake_first_data = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10 (0 has own test)
    fake_first_data = []
    for i in range(num_fake_first_data):
        fake_first_data.append(MagicMock)

    arg_data.append(fake_first_data)
    for i in range(pytest.gen.randint(0, 10)): # arbitrary, from 0 to 10 more data
        arg_data.append(MagicMock())
    
    cut = DataSource.__new__(DataSource)

    # Act
    cut.__init__(arg_data)

    # Assert
    assert cut.index == 0
    assert cut.data == arg_data
    assert cut.data_dimension == num_fake_first_data

def test__init__DataSource_sets_index_to_0_and_data_to_given_data_and_data_dimension_to_0_when_given_data_is_occupied_and_first_data_has_len_0():
    # Arrange
    arg_data = []

    fake_first_data = []

    arg_data.append(fake_first_data)
    for i in range(pytest.gen.randint(0, 10)): # arbitrary, from 0 to 10 more data
        arg_data.append(MagicMock())
    
    cut = DataSource.__new__(DataSource)

    # Act
    cut.__init__(arg_data)

    # Assert
    assert cut.index == 0
    assert cut.data == arg_data
    assert cut.data_dimension == 0

def test__init__DataSource_sets_index_to_0_and_data_to_given_data_and_data_dimension_to_0_when_given_data_is_vacant():
    # Arrange
    arg_data = []
    
    cut = DataSource.__new__(DataSource)

    # Act
    cut.__init__(arg_data)

    # Assert
    assert cut.index == 0
    assert cut.data == arg_data
    assert cut.data_dimension == 0

def test__init__DataSource_default_given_data_is_empty_list():
    # Arrange
    cut = DataSource.__new__(DataSource)

    # Act
    cut.__init__()

    # Assert
    assert cut.index == 0
    assert cut.data == []
    assert cut.data_dimension == 0

# get_next tests

# has_more tests

# has_data tests


# class TestDataSource(unittest.TestCase):

#     def setUp(self):
#         # self.test_path = os.path.dirname(os.path.abspath(__file__))
#         self.empty_D = DataSource()
#         self.nonempty_D = DataSource([['1'], ['2'], ['3']])

#     def test_init_empty_data_source(self):
#         self.assertEqual(self.empty_D.index, 0)
#         self.assertEqual(self.empty_D.data, [])
#         self.assertEqual(self.empty_D.data_dimension, 0)

#     def test_init_non_empty_data_source(self):
#         self.assertEqual(self.nonempty_D.index, 0)
#         self.assertEqual(self.nonempty_D.data, [['1'], ['2'], ['3']])
#         self.assertEqual(self.nonempty_D.data_dimension, 1)

#     def test_get_next(self):
#         next = self.nonempty_D.get_next()
#         self.assertEqual(self.nonempty_D.index, 1)
#         self.assertEqual(next, ['1'])

#     def test_has_more(self):
#         empty_answer = self.empty_D.has_more()
#         non_empty_answer = self.nonempty_D.has_more()

#         self.assertEqual(empty_answer, False)
#         self.assertEqual(non_empty_answer, True)

#     def test_has_data(self):
#         empty_answer = self.empty_D.has_more()
#         non_empty_answer = self.nonempty_D.has_more()
#         self.assertEqual(empty_answer, False)
#         self.assertEqual(non_empty_answer, True)

#         D = DataSource([['TIME', '-'], ['TIME', '-'], ['TIME', '-']])
#         answer = D.has_data()
#         self.assertEqual(answer, False)

#         D = DataSource([['TIME'], ['TIME'], ['TIME']])
#         answer = D.has_data()
#         self.assertEqual(answer, False)

#         D = DataSource([[], [], []])
#         answer = D.has_data()
#         self.assertEqual(answer, False)


#         D = DataSource()
#         answer = D.has_data()
#         self.assertEqual(answer, False)

# if __name__ == '__main__':
#     unittest.main()


