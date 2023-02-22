""" Test DataSource Functionality """
import pytest
from mock import MagicMock

import src.data_handling.data_source as data_source
from src.data_handling.data_source import DataSource

# __init__ tests
def test_DataSource__init__sets_index_to_0_and_data_to_given_data_and_data_dimension_to_len_of_first_data_in_data_when_given_data_occupied_and_first_data_has_len_more_than_0():
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

def test_DataSource__init__sets_index_to_0_and_data_to_given_data_and_data_dimension_to_0_when_given_data_is_occupied_and_first_data_has_len_0():
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

def test_DataSource__init__sets_index_to_0_and_data_to_given_data_and_data_dimension_to_0_when_given_data_is_vacant():
    # Arrange
    arg_data = []
    
    cut = DataSource.__new__(DataSource)

    # Act
    cut.__init__(arg_data)

    # Assert
    assert cut.index == 0
    assert cut.data == arg_data
    assert cut.data_dimension == 0

def test_DataSource__init__default_given_data_is_empty_list():
    # Arrange
    cut = DataSource.__new__(DataSource)

    # Act
    cut.__init__()

    # Assert
    assert cut.index == 0
    assert cut.data == []
    assert cut.data_dimension == 0

# get_next tests
def test_DataSource_get_next_increments_index_and_returns_data_at_index_minus_1_when_index_minus_1_is_less_than_len_data():
    # Arrange
    initial_index = pytest.gen.randint(0, 10) # arbitrary, from 0 to 10

    cut = DataSource.__new__(DataSource)
    cut.index = initial_index
    cut.data = []

    fake_num_data = cut.index + pytest.gen.randint(1, 5) # arbitrary, from 1 to 5 more data than index

    for i in range(fake_num_data):
        cut.data.append(MagicMock())

    # Act
    result = cut.get_next()

    # Assert
    assert cut.index == initial_index + 1
    assert result == cut.data[initial_index]

def test_DataSource_get_next_raises_Exception_when_data_is_vacant_but_still_increments_index():
    # Arrange
    initial_index = pytest.gen.randint(0, 10) # arbitrary, from 0 to 10

    cut = DataSource.__new__(DataSource)
    cut.index = initial_index
    cut.data = []

    # Act
    with pytest.raises(Exception) as e_info:
        cut.get_next()

    # Assert
    assert cut.index == initial_index + 1
    assert e_info.match('list index out of range')

def test_DataSource_get_next_raises_Exception_when_index_is_incremented_beyond_data_size():
    # Arrange
    initial_index = pytest.gen.randint(0, 10) # arbitrary, from 0 to 10

    cut = DataSource.__new__(DataSource)
    cut.index = initial_index
    cut.data = []

    fake_num_data = initial_index

    for i in range(fake_num_data):
        cut.data.append(MagicMock())

    # Act
    with pytest.raises(Exception) as e_info:
        cut.get_next()

    # Assert
    assert cut.index == initial_index + 1
    assert len(cut.data) == cut.index - 1
    assert e_info.match('list index out of range')

# has_more tests
def test_DataSource_has_more_returns_True_when_index_is_less_than_data_len(mocker):
    # Arrange
    cut = DataSource.__new__(DataSource)
    cut.index = pytest.gen.randint(0, 5) # arbitrary, from 0 to 5
    cut.data = MagicMock()

    mocker.patch('src.data_handling.data_source.len', return_value=cut.index+pytest.gen.randint(1, 100)) # arbitrary, from 1 to 100 more data than index

    # Act
    result = cut.has_more()

    # Assert
    assert data_source.len.call_count == 1
    assert data_source.len.call_args_list[0].args == (cut.data, )
    assert result == True

def test_DataSource_has_more_returns_False_when_index_eq_data_len(mocker):
    # Arrange
    cut = DataSource.__new__(DataSource)
    cut.index = pytest.gen.randint(0, 5) # arbitrary, from 0 to 5
    cut.data = MagicMock()

    mocker.patch('src.data_handling.data_source.len', return_value=cut.index)

    # Act
    result = cut.has_more()

    # Assert
    assert data_source.len.call_count == 1
    assert data_source.len.call_args_list[0].args == (cut.data, )
    assert result == False

def test_DataSource_has_more_returns_False_when_index_greater_than_data_len(mocker):
    # Arrange
    cut = DataSource.__new__(DataSource)
    cut.index = pytest.gen.randint(1, 5) # arbitrary, from 1 to 5 (min 1 to be able to subtract)
    cut.data = MagicMock()

    mocker.patch('src.data_handling.data_source.len', return_value=cut.index-1)

    # Act
    result = cut.has_more()

    # Assert
    assert data_source.len.call_count == 1
    assert data_source.len.call_args_list[0].args == (cut.data, )
    assert result == False

# has_data tests
def test_DataSource_has_data_returns_False_when_data_is_empty_list():
    # Arrange
    cut = DataSource.__new__(DataSource)
    cut.data = []

    # Act
    result = cut.has_data()

    # Assert
    assert result == False

def test_DataSource_has_data_returns_False_when_all_data_points_items_after_timestamp_are_empty_steps():
    # Arrange
    cut = DataSource.__new__(DataSource)
    cut.data_dimension = pytest.gen.randint(2, 10) # arbitrary, from 2 to 10 data size, 2 accounts for 1 as timestamp
    cut.data = []
    for i in range(pytest.gen.randint(1, 10)): # arbitrary, from 1 to 10 data points
        fake_data_pt = ['-'] * cut.data_dimension
        fake_timestamp = MagicMock()
        fake_data_pt[0] = fake_timestamp
        cut.data.append(fake_data_pt)

    # Act
    result = cut.has_data()

    # Assert
    assert result == False

def test_DataSource_has_data_returns_True_when_at_least_one_data_point_has_non_empty_step_after_timestamp():
    # Arrange
    cut = DataSource.__new__(DataSource)
    cut.data_dimension = pytest.gen.randint(2, 10) # arbitrary, from 2 to 10 data size, 2 accounts for 1 as timestamp
    cut.data = []
    for i in range(pytest.gen.randint(1, 10)): # arbitrary, from 1 to 10 data points
        fake_data_pt = ['-'] * cut.data_dimension
        fake_timestamp = MagicMock()
        fake_data_pt[0] = fake_timestamp
        cut.data.append(fake_data_pt)

    cut.data[pytest.gen.randint(0, len(cut.data) - 1)][pytest.gen.randint(1, cut.data_dimension - 1)] = MagicMock() # [from 0th data pt up to last data pt][from first item after timestamp to last item]

    # Act
    result = cut.has_data()

    # Assert
    assert result == True

def test_DataSource_has_data_returns_True_when_at_least_one_data_point_has_different_size_data_than_size_dimension():
    # Arrange
    cut = DataSource.__new__(DataSource)
    cut.data_dimension = pytest.gen.randint(2, 10) # arbitrary, from 2 to 10 data size, 2 accounts for 1 as timestamp
    cut.data = []
    for i in range(pytest.gen.randint(1, 10)): # arbitrary, from 1 to 10 data points
        fake_data_pt = ['-'] * cut.data_dimension
        fake_timestamp = MagicMock()
        fake_data_pt[0] = fake_timestamp
        cut.data.append(fake_data_pt)

    if pytest.gen.randint(0, 1):
        random_non_dimension_size = pytest.gen.randint(0, cut.data_dimension - 2) # less than
    else:
        random_non_dimension_size = pytest.gen.randint(cut.data_dimension, cut.data_dimension + 10) # greater than
    cut.data[pytest.gen.randint(0, len(cut.data) - 1)] = [MagicMock()] + ['-'] * random_non_dimension_size # [from 0th data pt up to last data pt], replace random data pt with greater or less than expected dimension

    # Act
    result = cut.has_data()

    # Assert
    assert result == True
