""" Test Time Sync Functionality """
import os
import unittest
import pytest

from mock import MagicMock
import data_handling.time_synchronizer as time_synchronizer
from data_handling.time_synchronizer import TimeSynchronizer

# __init__ tests
def test_TimeSynchronizer_init_does_not_set_instance_default_values_when_calls_to_init_sync_data_and_sort_data_do_not_raise_exceptions(mocker):
    # Arrange
    arg_headers = MagicMock()
    arg_dataFrames = MagicMock()
    arg_test_configs = MagicMock()

    cut = TimeSynchronizer.__new__(TimeSynchronizer)

    mocker.patch.object(cut, 'init_sync_data')
    mocker.patch.object(cut, 'sort_data')

    # Act
    cut.__init__(arg_headers, arg_dataFrames, arg_test_configs)

    # Assert
    assert cut.init_sync_data.call_count == 1
    assert cut.init_sync_data.call_args_list[0].args == (arg_headers, arg_test_configs)
    assert cut.sort_data.call_count == 1
    assert cut.sort_data.call_args_list[0].args == (arg_dataFrames, )
    assert hasattr(cut, 'ordered_sources') == False
    assert hasattr(cut, 'ordered_fused_headers') == False
    assert hasattr(cut, 'ordered_fused_tests') == False
    assert hasattr(cut, 'indices_to_remove') == False
    assert hasattr(cut, 'offsets') == False
    assert hasattr(cut, 'sim_data') == False

def test_TimeSynchronizer_init_sets_instance_default_values_when_call_to_init_sync_data_raises_exception(mocker):
    # Arrange
    arg_headers = MagicMock()
    arg_dataFrames = MagicMock()
    arg_test_configs = MagicMock()

    cut = TimeSynchronizer.__new__(TimeSynchronizer)

    mocker.patch.object(cut, 'init_sync_data', side_effect=Exception())
    mocker.patch.object(cut, 'sort_data')

    # Act
    cut.__init__(arg_headers, arg_dataFrames, arg_test_configs)

    # Assert
    assert cut.init_sync_data.call_count == 1
    assert cut.init_sync_data.call_args_list[0].args == (arg_headers, arg_test_configs)
    assert cut.sort_data.call_count == 0
    assert cut.ordered_sources == []
    assert cut.ordered_fused_headers == []
    assert cut.ordered_fused_tests == []
    assert cut.indices_to_remove == []
    assert cut.offsets == {}
    assert cut.sim_data == []

def test_TimeSynchronizer_init_sets_instance_default_values_when_call_to_sort_data_raises_exception(mocker):
    # Arrange
    arg_headers = MagicMock()
    arg_dataFrames = MagicMock()
    arg_test_configs = MagicMock()

    cut = TimeSynchronizer.__new__(TimeSynchronizer)

    mocker.patch.object(cut, 'init_sync_data')
    mocker.patch.object(cut, 'sort_data', side_effect=Exception())

    # Act
    cut.__init__(arg_headers, arg_dataFrames, arg_test_configs)

    # Assert
    assert cut.init_sync_data.call_count == 1
    assert cut.init_sync_data.call_args_list[0].args == (arg_headers, arg_test_configs)
    assert cut.sort_data.call_count == 1
    assert cut.sort_data.call_args_list[0].args == (arg_dataFrames, )
    assert cut.ordered_sources == []
    assert cut.ordered_fused_headers == []
    assert cut.ordered_fused_tests == []
    assert cut.indices_to_remove == []
    assert cut.offsets == {}
    assert cut.sim_data == []

# init_sync_data tests
def test_TimeSynchronizer_init_sync_data_raises_Exception_when_given_headers_is_empty_dict():
    # Arrange
    arg_headers = {}
    arg_configs = MagicMock()

    cut = TimeSynchronizer.__new__(TimeSynchronizer)

    # Act
    with pytest.raises(Exception) as e_info:
        cut.init_sync_data(arg_headers, arg_configs)

    # Assert
    assert e_info.match('Unable to initialize sync data: Empty Dataset')

def test_TimeSynchronizer_init_sync_data_raises_Exception_when_given_configs_is_empty_dict():
    # Arrange
    arg_headers = MagicMock()
    arg_configs = {}

    cut = TimeSynchronizer.__new__(TimeSynchronizer)

    # Act
    with pytest.raises(Exception) as e_info:
        cut.init_sync_data(arg_headers, arg_configs)

    # Assert
    assert e_info.match('Unable to initialize sync data: Empty Dataset')
    
def test_TimeSynchronizer_init_sync_data_raises_Exception_when_given_headers_or_configs_is_empty_dict():
    # Arrange
    arg_headers = {} if pytest.gen.randint(0, 1) else MagicMock()
    arg_configs = {} if arg_headers != {} else MagicMock()

    cut = TimeSynchronizer.__new__(TimeSynchronizer)

    # Act
    with pytest.raises(Exception) as e_info:
        cut.init_sync_data(arg_headers, arg_configs)

    # Assert
    if arg_headers == {}:
        assert arg_configs != {}
    else:
        assert arg_configs == {}
    assert e_info.match('Unable to initialize sync data: Empty Dataset')

# NOTE: test_TimeSynchronizer_init_sync_data_sets_instance_variables_to_the_expected_values is not my favorite name because it isn't very descriptive, but had I described everything happening in this function the name would be ENORMOUS
def test_TimeSynchronizer_init_sync_data_sets_instance_variables_to_the_expected_values(mocker):
    # Arrange
    arg_headers = {}
    arg_configs = {}
    arg_configs['test_assignments'] = {}

    num_fake_headers = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10 header dict entries (0 has own test)
    fake_time_indices_for_removal = MagicMock()
    fake_ordered_fused_headers = [MagicMock()]
    fake_ordered_fused_tests = [MagicMock()]
    fake_unclean_fused_tests = list(range(num_fake_headers)) + list(range(num_fake_headers))
    fake_unclean_fused_tests.sort()

    expected_ordered_fused_tests = [[['SYNC', 'TIME']], fake_ordered_fused_tests[0]]
    expected_ordered_fused_headers = ['TIME', fake_ordered_fused_headers[0]]
    expected_start_indices = {}

    for i in range(num_fake_headers):
        fake_source = MagicMock()
        arg_headers[fake_source] = [i, i]
        arg_configs['test_assignments'][fake_source] = [i, i]
        expected_start_indices[fake_source] = i * 2 # to show offsets change by length of arg_headers[fake_source]

    cut = TimeSynchronizer.__new__(TimeSynchronizer)

    mocker.patch.object(cut, 'remove_time_headers', return_value=(fake_time_indices_for_removal, fake_ordered_fused_headers))
    mocker.patch.object(cut, 'remove_time_datapoints', return_value=(fake_ordered_fused_tests))

    # Act
    cut.init_sync_data(arg_headers, arg_configs)

    # Assert
    assert cut.remove_time_headers.call_count == 1
    assert cut.remove_time_headers.call_args_list[0].args == (fake_unclean_fused_tests, )
    assert cut.remove_time_datapoints.call_count == 1
    assert cut.remove_time_datapoints.call_args_list[0].args == (fake_unclean_fused_tests, fake_time_indices_for_removal)
    assert cut.ordered_sources == list(arg_headers.keys())
    assert cut.ordered_fused_tests == expected_ordered_fused_tests
    assert cut.ordered_fused_headers == expected_ordered_fused_headers
    assert cut.indices_to_remove == fake_time_indices_for_removal
    assert cut.offsets == expected_start_indices

# sort_data tests
def test_TimeSynchronizer_sort_data_sets_instance_sim_data_to_empty_list_when_given_dataFrames_is_empty_dict():
    # Arrange
    arg_dataFrames = {}

    cut = TimeSynchronizer.__new__(TimeSynchronizer)
    # NOTE: this accessed item in the cut helps set num_sources, but that is an usued variable that should be removed, after it is setting cut.ordered_sources no longer needs to be done
    cut.ordered_sources = MagicMock()

    # Act
    cut.sort_data(arg_dataFrames)

    # Assert
    assert cut.sim_data == []

def test_TimeSynchronizer_sort_data_sets_instance_sim_data_to_list_of_a_single_return_from_initialize_clean_data_array_when_arg_dataFrames_contains_one_element(mocker):
    # Arrange
    fake_dataFrame_key = MagicMock()
    fake_data = MagicMock()
    fake_clean_data_array = MagicMock()

    arg_dataFrames = {fake_dataFrame_key:fake_data}

    cut = TimeSynchronizer.__new__(TimeSynchronizer)

    mocker.patch.object(cut, 'initialize_clean_data_array', return_value=fake_clean_data_array)

    # Act
    cut.sort_data(arg_dataFrames)

    # Assert
    assert cut.initialize_clean_data_array.call_count == 1
    assert cut.initialize_clean_data_array.call_args_list[0].args == (arg_dataFrames, fake_dataFrame_key)
    assert cut.sim_data == [fake_clean_data_array]

def test_TimeSynchronizer_sort_data_sets_instance_sim_data_to_list_of_returns_from_initialize_clean_data_array_when_arg_dataFrames_contains_multiple_elements(mocker):
    # Arrange
    num_elems = pytest.gen.randint(2, 10) # arbitrary, from 2 to 10 (0 and 1 have own tests)
    expected_dataFrame_keys = []
    fake_clean_data_arrays = []
    arg_dataFrames = {}
    for i in range(num_elems):
        key = str(MagicMock())
        data = MagicMock()
        clean_data_array = MagicMock()
        expected_dataFrame_keys.append(key)
        fake_clean_data_arrays.append(clean_data_array)
        arg_dataFrames[key] = data
    expected_dataFrame_keys.sort()

    cut = TimeSynchronizer.__new__(TimeSynchronizer)

    mocker.patch.object(cut, 'initialize_clean_data_array', side_effect=fake_clean_data_arrays)

    # Act
    cut.sort_data(arg_dataFrames)

    # Assert
    assert cut.initialize_clean_data_array.call_count == num_elems
    for i in range(num_elems):
        assert cut.initialize_clean_data_array.call_args_list[i].args == (arg_dataFrames, expected_dataFrame_keys[i])
    assert cut.sim_data == fake_clean_data_arrays

# initialize_clean_data_array tests
def test_TimeSynchronizer_initialize_clean_data_array_calls_remove_time_datapoints_and_returns_expected_list_when_dataFrames_time_is_empty(mocker):
    # Arrange
    arg_time = MagicMock()
    arg_dataFrames = {arg_time : {}}

    rand_num_indicies = pytest.gen.randint(0, 10) # arbitrary, from 0 to 10
    fake_indices_to_remove = [MagicMock()] * rand_num_indicies
    rand_num_headers = pytest.gen.randint(0, 10) # arbitrary, from 0 to 10
    fake_ordered_fused_headers = [MagicMock()] * rand_num_headers

    cut = TimeSynchronizer.__new__(TimeSynchronizer)
    cut.indices_to_remove = fake_indices_to_remove
    cut.ordered_fused_headers = fake_ordered_fused_headers

    forced_return_deepcopy = MagicMock()
    mocker.patch('data_handling.time_synchronizer.copy.deepcopy', return_value=forced_return_deepcopy)
    mocker.patch.object(cut, 'remove_time_datapoints')
    mocker.patch.object(cut, 'copy_to_with_offset')

    expected_result = [arg_time] + ['-'] * (rand_num_indicies + rand_num_headers - 1)

    # Act
    result = cut.initialize_clean_data_array(arg_dataFrames, arg_time)

    # Assert
    assert cut.copy_to_with_offset.call_count == 0
    assert time_synchronizer.copy.deepcopy.call_count == 1
    assert time_synchronizer.copy.deepcopy.call_args_list[0].args == (fake_indices_to_remove, )
    assert cut.remove_time_datapoints.call_count == 1
    assert cut.remove_time_datapoints.call_args_list[0].args == (expected_result, forced_return_deepcopy)
    assert result == expected_result

def test_TimeSynchronizer_initialize_clean_data_array_loop_makes_expected_call_to_copy_to_with_offset_and_calls_remove_time_datapoints_and_returns_expected_list_when_dataFrames_time_contains_one_element(mocker):
    # Arrange
    rand_num_indicies = pytest.gen.randint(0, 10) # arbitrary, from 0 to 10
    fake_indices_to_remove = [MagicMock()] * rand_num_indicies
    rand_num_headers = pytest.gen.randint(0, 10) # arbitrary, from 0 to 10
    fake_ordered_fused_headers = [MagicMock()] * rand_num_headers
    fake_key = MagicMock()
    fake_data = MagicMock()
    fake_offset = MagicMock()
    fake_offsets = {fake_key : fake_offset}

    arg_time = MagicMock()
    arg_dataFrames = {arg_time : {fake_key : fake_data}}

    cut = TimeSynchronizer.__new__(TimeSynchronizer)
    cut.indices_to_remove = fake_indices_to_remove
    cut.ordered_fused_headers = fake_ordered_fused_headers
    cut.offsets = fake_offsets

    forced_return_deepcopy = MagicMock()
    mocker.patch('data_handling.time_synchronizer.copy.deepcopy', return_value=forced_return_deepcopy)
    mocker.patch.object(cut, 'remove_time_datapoints')
    mocker.patch.object(cut, 'copy_to_with_offset')

    expected_result = [arg_time] + ['-'] * (rand_num_indicies + rand_num_headers - 1)

    # Act
    result = cut.initialize_clean_data_array(arg_dataFrames, arg_time)

    # Assert
    assert cut.copy_to_with_offset.call_count == 1
    assert cut.copy_to_with_offset.call_args_list[0].args == (expected_result, fake_data, fake_offset)
    assert time_synchronizer.copy.deepcopy.call_count == 1
    assert time_synchronizer.copy.deepcopy.call_args_list[0].args == (fake_indices_to_remove, )
    assert cut.remove_time_datapoints.call_count == 1
    assert cut.remove_time_datapoints.call_args_list[0].args == (expected_result, forced_return_deepcopy)
    assert result == expected_result

def test_TimeSynchronizer_initialize_clean_data_array_loop_makes_expected_call_to_copy_to_with_offset_and_calls_remove_time_datapoints_and_returns_expected_list_when_dataFrames_time_contains_many_elements(mocker):
    # Arrange
    rand_num_indicies = pytest.gen.randint(0, 10) # arbitrary, from 0 to 10
    fake_indices_to_remove = [MagicMock()] * rand_num_indicies
    rand_num_headers = pytest.gen.randint(0, 10) # arbitrary, from 0 to 10
    fake_ordered_fused_headers = [MagicMock()] * rand_num_headers
    num_elems_dataFrames = pytest.gen.randint(2, 10) # arbitrary, from 2 to 10
    fake_key = []
    fake_data = []
    fake_offset = []
    fake_offsets = {}
    fake_frame = {}
    for i in range(num_elems_dataFrames):
        fake_key.append(MagicMock())
        fake_data.append(MagicMock())
        fake_offset.append(MagicMock())
        fake_offsets[fake_key[i]] = fake_offset[i]
        fake_frame[fake_key[i]] = fake_data[i]

    arg_time = MagicMock()
    arg_dataFrames = {arg_time : fake_frame}

    cut = TimeSynchronizer.__new__(TimeSynchronizer)
    cut.indices_to_remove = fake_indices_to_remove
    cut.ordered_fused_headers = fake_ordered_fused_headers
    cut.offsets = fake_offsets

    forced_return_deepcopy = MagicMock()
    mocker.patch('data_handling.time_synchronizer.copy.deepcopy', return_value=forced_return_deepcopy)
    mocker.patch.object(cut, 'remove_time_datapoints')
    mocker.patch.object(cut, 'copy_to_with_offset')

    expected_result = [arg_time] + ['-'] * (rand_num_indicies + rand_num_headers - 1)

    # Act
    result = cut.initialize_clean_data_array(arg_dataFrames, arg_time)

    # Assert
    assert cut.copy_to_with_offset.call_count == num_elems_dataFrames
    for i in range(num_elems_dataFrames):
        assert cut.copy_to_with_offset.call_args_list[i].args == (expected_result, fake_data[i], fake_offset[i])
    assert time_synchronizer.copy.deepcopy.call_count == 1
    assert time_synchronizer.copy.deepcopy.call_args_list[0].args == (fake_indices_to_remove, )
    assert cut.remove_time_datapoints.call_count == 1
    assert cut.remove_time_datapoints.call_args_list[0].args == (expected_result, forced_return_deepcopy)
    assert result == expected_result

# copy_to_with_offset tests
def test_TimeSynchronizer_copy_to_with_offset_does_nothing_when_src_array_is_empty():
    # Arrange
    arg_src_array = []
    arg_offset = pytest.gen.randint(0, 10) # arbitray, from 0 to 10
    len_dest_array = pytest.gen.randint(0, 10) # arbitrary, from 0 to 10
    arg_dest_array = [MagicMock()] * len_dest_array

    cut = TimeSynchronizer.__new__(TimeSynchronizer)

    src_array_copy = arg_src_array.copy()
    dest_array_copy = arg_dest_array.copy()

    # Act
    cut.copy_to_with_offset(arg_dest_array, arg_src_array, arg_offset)

    # Assert
    assert arg_src_array == src_array_copy
    assert arg_dest_array == dest_array_copy

def test_TimeSynchronizer_copy_to_with_offset_raises_error_when_len_dest_array_less_than_sum_of_len_src_array_and_offset():
    # Arrange
    len_src_array = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10
    arg_src_array = [MagicMock()] * len_src_array
    arg_offset = pytest.gen.randint(0, 10) # arbitray, from 0 to 10
    len_dest_array = pytest.gen.randint(0, len_src_array + arg_offset - 1) # arbitrary, at most size of src_array + offset - 1
    arg_dest_array = [MagicMock()] * len_dest_array

    cut = TimeSynchronizer.__new__(TimeSynchronizer)

    # Act
    with pytest.raises(IndexError) as e_info:
        cut.copy_to_with_offset(arg_dest_array, arg_src_array, arg_offset)

    # Assert
    assert e_info.match('list assignment index out of range')

def test_TimeSynchronizer_copy_to_with_offset_sets_expected_indices_in_arg_dest_array_to_expected_values_from_arg_src_array_according_to_arg_offset():
    # Arrange
    len_src_array = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10
    arg_src_array = [MagicMock()] * len_src_array
    arg_offset = pytest.gen.randint(0, 10) # arbitray, from 0 to 10
    len_dest_array = pytest.gen.randint(len_src_array + arg_offset, 50) # arbitrary, at least size of src_array + offset
    arg_dest_array = [MagicMock()] * len_dest_array

    cut = TimeSynchronizer.__new__(TimeSynchronizer)

    src_array_copy = arg_src_array.copy()
    dest_array_copy = arg_dest_array.copy()
    expected_dest_array = arg_dest_array.copy()
    for i in range(arg_offset, arg_offset + len_src_array):
        expected_dest_array[i] = arg_src_array[i - arg_offset]

    # Act
    cut.copy_to_with_offset(arg_dest_array, arg_src_array, arg_offset)

    # Assert
    assert arg_src_array == src_array_copy
    assert arg_dest_array != dest_array_copy
    assert arg_dest_array == expected_dest_array

# TODO: coverage does not require loops be done more than once, but it is good practice; however, this is a complex endeavor here and is should be done at a later time and/or refactored for easier testing
# def test_TimeSynchronizer_sort_data_sets_multisource(mocker):
#     # Arrange
#     fake_dataFrame_key = MagicMock()
#     fake_data_key = MagicMock()
#     fake_data_key2 = MagicMock()
#     fake_source_data = []
#     fake_source_data2 = [MagicMock()]
#     arg_dataFrames = {fake_dataFrame_key:{fake_data_key:fake_source_data, fake_data_key2:fake_source_data2}}

#     fake_deep_copy = MagicMock()

#     mocker.patch('data_handling.time_synchronizer.copy.deepcopy', return_value=fake_deep_copy)

#     expected_clean_data_symbol = '-'

#     cut = TimeSynchronizer.__new__(TimeSynchronizer)
#     # NOTE: this accessed item in the cut helps set num_sources, but that is an usued variable that should be removed, after it is setting cut.ordered_sources no longer needs to be done
#     cut.ordered_sources = MagicMock()
#     num_fake_total = pytest.gen.randint(3, 20) # from 3, min required, to arbitrary 20
#     num_fake_indices_to_remove = pytest.gen.randint(0, num_fake_total)
#     num_fake_order_fused_headers = num_fake_total - num_fake_indices_to_remove
#     cut.indices_to_remove = []
#     for i in range(num_fake_indices_to_remove):
#         cut.indices_to_remove.append(MagicMock())
#     cut.ordered_fused_headers = []
#     for i in range(num_fake_order_fused_headers):
#         cut.ordered_fused_headers.append(MagicMock())
#     num_fake_source_data = pytest.gen.randint(1, num_fake_total) - 1 # arbitrary, from 1 to total items expected (0 has own test)
#     for i in range(num_fake_source_data):
#         fake_source_data.append(MagicMock())
#     fake_offset = pytest.gen.randint(0, (num_fake_total - num_fake_source_data - 1)) # slightly arbitrary, allowing room for all source data between 0 and total items minus 1 for proper indexing
#     cut.offsets = {fake_data_key:fake_offset, fake_data_key2:fake_offset}

#     num_prior_clean_data_symbols = [expected_clean_data_symbol]*(num_fake_total+(fake_offset-num_fake_total))
#     num_remaining_clean_data_symbols = [expected_clean_data_symbol]*(num_fake_total-(fake_offset+num_fake_source_data+1))
#     expected_data = num_prior_clean_data_symbols + fake_source_data + num_remaining_clean_data_symbols
#     # NOTE: verify_clean_array_before_alteration is necessary because in the Assert phase of test the value passed in is altered afterwards by the cut and cannot be verified as sent in the state checked here
#     def verify_clean_array_before_alteration(data, indices_to_remove):
#         assert data == expected_data
#         assert indices_to_remove == fake_deep_copy

#     mocker.patch.object(cut, 'remove_time_datapoints', side_effect=verify_clean_array_before_alteration)

#     # Act
#     cut.sort_data(arg_dataFrames)

#     # Assert
#     assert time_synchronizer.copy.deepcopy.call_count == 1
#     assert time_synchronizer.copy.deepcopy.call_args_list[0].args == (cut.indices_to_remove, )
#     assert cut.remove_time_datapoints.call_count == 1
#     #assert cut.remove_time_datapoints.call_args_list[0].args == ([], fake_deep_copy), does not work due to first item alteration
#     assert cut.sim_data == [[fake_dataFrame_key] + expected_data]

# remove_time_headers tests
def test_TimeSynchronizer_remove_time_headers_retuns_tuple_of_empty_list_and_empty_list_when_given_hdrs_list_is_vacant():
    # Arrange
    arg_hdrs_list = []

    cut = TimeSynchronizer.__new__(TimeSynchronizer)

    # Act
    result = cut.remove_time_headers(arg_hdrs_list)

    # Assert
    assert result == ([], [])

def test_TimeSynchronizer_remove_time_headers_retuns_tuple_of_empty_list_and_copy_of_given_hdrs_list_when_given_hdrs_list_has_no_str_TIME_or_str_time():
    # Arrange
    arg_hdrs_list = []

    expected_clean_hdrs_list = []
    num_fake_hdrs = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10 (0 has own test)
    for i in range(num_fake_hdrs):
        fake_hdr = MagicMock()
        arg_hdrs_list.append(fake_hdr)
        expected_clean_hdrs_list.append(fake_hdr)

    cut = TimeSynchronizer.__new__(TimeSynchronizer)

    # Act
    result = cut.remove_time_headers(arg_hdrs_list)

    # Assert
    assert result == ([], expected_clean_hdrs_list)
    assert result[1] is not arg_hdrs_list

def test_TimeSynchronizer_remove_time_headers_retuns_tuple_of_list_of_single_location_and_and_copy_of_given_hdrs_list_with_time_hdr_removed_when_given_hdrs_list_has_single_time_hdr():
    # Arrange
    arg_hdrs_list = []

    expected_clean_hdrs_list = []
    num_fake_hdrs = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10 (0 has own test)
    fake_time_index = pytest.gen.randint(0, num_fake_hdrs-1)
    fake_time_hdr = 'TIME' if pytest.gen.randint(0,1) else 'time'
    for i in range(num_fake_hdrs):
        fake_hdr = MagicMock()
        if i != fake_time_index:
            arg_hdrs_list.append(fake_hdr)
            expected_clean_hdrs_list.append(fake_hdr)
        else:
            arg_hdrs_list.append(fake_time_hdr)

    cut = TimeSynchronizer.__new__(TimeSynchronizer)

    # Act
    result = cut.remove_time_headers(arg_hdrs_list)

    # Assert
    assert result == ([fake_time_index], expected_clean_hdrs_list)
    assert result[1] is not arg_hdrs_list
    assert result[1] == [hdr for hdr in arg_hdrs_list if hdr != fake_time_hdr]

def test_TimeSynchronizer_remove_time_headers_retuns_tuple_of_sorted_list_of_all_indices_of_time_hdr_and_and_copy_of_given_hdrs_list_with_time_hdrs_removed_when_given_hdrs_list_has_any_number_of_time_headers():
    # Arrange
    arg_hdrs_list = []

    expected_clean_hdrs_list = []
    num_fake_hdrs = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10 (0 has own test)
    fake_time_indices = pytest.gen.sample(list(range(num_fake_hdrs)), pytest.gen.randint(0, num_fake_hdrs)) # list of indices from none up to all of them
    fake_time_indices.sort()
    for i in range(num_fake_hdrs):
        fake_hdr = MagicMock()
        if not fake_time_indices.count(i):
            arg_hdrs_list.append(fake_hdr)
            expected_clean_hdrs_list.append(fake_hdr)
        else:
            arg_hdrs_list.append('TIME' if pytest.gen.randint(0,1) else 'time')

    cut = TimeSynchronizer.__new__(TimeSynchronizer)

    # Act
    result = cut.remove_time_headers(arg_hdrs_list)

    # Assert
    assert result == (fake_time_indices, expected_clean_hdrs_list)
    assert result[0] is not fake_time_indices
    assert result[1] is not arg_hdrs_list
    assert result[1] is not expected_clean_hdrs_list
    assert result[1] == [hdr for hdr in arg_hdrs_list if hdr != 'TIME' and hdr != 'time']

# remove_time_datapoints tests
def test_TimeSynchronizer_remove_time_datapoints_returns_given_data_when_given_indices_to_remove_is_vacant():
    # Arrange
    arg_data = []
    arg_indices_to_remove = []

    expected_result = []
    num_fake_data = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10 (0 has own test)

    for i in range(num_fake_data):
        fake_data = MagicMock()
        arg_data.append(fake_data)
        expected_result.append(fake_data)

    cut = TimeSynchronizer.__new__(TimeSynchronizer)

    # Act
    result = cut.remove_time_datapoints(arg_data, arg_indices_to_remove)

    # Assert
    assert result == expected_result

def test_TimeSynchronizer_remove_time_datapoints_returns_empty_list_when_given_data_is_size_of_1_and_given_indices_to_remove_is_list_with_0_as_only_item():
    # Arrange
    arg_data = [MagicMock()]
    arg_indices_to_remove = [0]

    cut = TimeSynchronizer.__new__(TimeSynchronizer)

    # Act
    result = cut.remove_time_datapoints(arg_data, arg_indices_to_remove)

    # Assert
    assert result == []

def test_TimeSynchronizer_remove_time_datapoints_raises_IndexError_when_given_data_does_not_have_an_index_in_given_indices_to_remove():
    # Arrange
    arg_data = []
    arg_indices_to_remove = [1]

    cut = TimeSynchronizer.__new__(TimeSynchronizer)

    # Act
    with pytest.raises(IndexError) as e_info:  
        cut.remove_time_datapoints(arg_data, arg_indices_to_remove)

    # Assert
    assert e_info.match("list assignment index out of range")

def test_TimeSynchronizer_remove_time_datapoints_returns_data_with_indices_removed_that_exist_in_given_indices_to_remove():
    # Arrange
    arg_data = []
    arg_indices_to_remove = []

    expected_result = []
    num_fake_data = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10 (0 has own test)
    fake_indices_to_remove = pytest.gen.sample(list(range(num_fake_data)), pytest.gen.randint(1, num_fake_data)) # make a randomized list from 1 index up to all the data indices (0 has own test)

    for i in range(num_fake_data):
        fake_data = MagicMock()
        arg_data.append(fake_data)
        if fake_indices_to_remove.count(i):
            arg_indices_to_remove.append(i)
        else:
            expected_result.append(fake_data)

    cut = TimeSynchronizer.__new__(TimeSynchronizer)

    # Act
    result = cut.remove_time_datapoints(arg_data, arg_indices_to_remove)
    
    # Assert
    assert result == expected_result

def test_TimeSynchronizer_remove_time_datapoints_does_not_alter_given_indecies_to_remove():
    # Arrange
    arg_data = []
    arg_indices_to_remove = []

    expected_result = []
    num_fake_data = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10 (0 has own test)
    fake_indices_to_remove = pytest.gen.sample(list(range(num_fake_data)), pytest.gen.randint(1, num_fake_data)) # make a randomized list from 1 index up to all the data indices (0 has own test)

    for i in range(num_fake_data):
        fake_data = MagicMock()
        arg_data.append(fake_data)
        if fake_indices_to_remove.count(i):
            arg_indices_to_remove.append(i)
        else:
            expected_result.append(fake_data)

    cut = TimeSynchronizer.__new__(TimeSynchronizer)

    # Act
    result = cut.remove_time_datapoints(arg_data, arg_indices_to_remove)
    
    # Assert
    assert result == expected_result
    assert len(result) == num_fake_data - len(fake_indices_to_remove)
    assert len(arg_indices_to_remove) == len(fake_indices_to_remove)

# get_vehicle_metadata tests
def test_TimeSynchronizer_get_vehicle_metadata_returns_tuple_of_instance_values_ordered_fused_headers_and_ordered_fused_tests():
    # Arrange
    fake_ordered_fused_headers = MagicMock()
    fake_ordered_fused_tests = MagicMock()

    cut = TimeSynchronizer.__new__(TimeSynchronizer)
    cut.ordered_fused_headers = fake_ordered_fused_headers
    cut.ordered_fused_tests = fake_ordered_fused_tests

    # Act
    result = cut.get_vehicle_metadata()

    # Assert
    assert result == (fake_ordered_fused_headers, fake_ordered_fused_tests)

# get_sim_data tests
def test_TimeSynchronizer_get_sim_data_returns_instance_value_of_sim_data():
    # Arrange
    fake_sim_data = MagicMock()

    cut = TimeSynchronizer.__new__(TimeSynchronizer)
    cut.sim_data = fake_sim_data

    # Act
    result = cut.get_sim_data()

    # Assert
    assert result == fake_sim_data

# class TestTimeSynchronizer(unittest.TestCase):

#     def setUp(self):
#         self.test_path = os.path.dirname(os.path.abspath(__file__))
#         self.TS = TimeSynchronizer()

#     def test_init_empty_sync_data(self):
#         self.assertEqual(self.TS.ordered_sources, [])
#         self.assertEqual(self.TS.ordered_fused_headers, [])
#         self.assertEqual(self.TS.ordered_fused_tests, [])
#         self.assertEqual(self.TS.indices_to_remove, [])
#         self.assertEqual(self.TS.offsets, {})
#         self.assertEqual(self.TS.sim_data, [])

#     def test_init_sync_data(self):
#         hdrs = {'test_sample_01' : ['TIME', 'hdr_A', 'hdr_B'],
#                 'test_sample_02' : ['TIME', 'hdr_C']}        
        
#         # Even if you give configs with ss assignments, they should not be here at the binner stage 
#         configs = {'test_assignments': {'test_sample_01': [[['SYNC', 'TIME']], [['NOOP']], [['NOOP']]],
#                                         'test_sample_02': [[['SYNC', 'TIME']], [['NOOP']]]}, 
#                    'description_assignments': {'test_sample_01': ['Time', 'No description', 'No description']}}

#         self.TS.init_sync_data(hdrs, configs) 

#         self.assertEqual(self.TS.ordered_fused_tests, [[['SYNC', 'TIME']], [['NOOP']], [['NOOP']], [['NOOP']]])
#         self.assertEqual(self.TS.ordered_sources, ['test_sample_01', 'test_sample_02'])
#         self.assertEqual(self.TS.ordered_fused_headers, ['TIME', 'hdr_A', 'hdr_B', 'hdr_C'])
#         self.assertEqual(self.TS.indices_to_remove, [0,3])
#         self.assertEqual(self.TS.offsets, {'test_sample_01': 0, 'test_sample_02': 3})
#         self.assertEqual(self.TS.sim_data, [])

#     def test_sort_data(self):

#         self.TS.ordered_fused_tests = [[['SYNC', 'TIME']], [['NOOP']], [['NOOP']], [['NOOP']]]
#         self.TS.ordered_sources = ['test_sample_01', 'test_sample_02']
#         self.TS.ordered_fused_headers = ['TIME', 'hdr_A', 'hdr_B', 'hdr_C']
#         self.TS.indices_to_remove =[0,3]
#         self.TS.offsets = {'test_sample_01': 0, 'test_sample_02': 3}
#         self.TS.unclean_fused_hdrs = ['TIME', 'hdr_A', 'hdr_B', 'TIME', 'hdr_C']

#         data = {'1234' : {'test_sample_01' : ['1234','202','0.3'],
#                           'test_sample_02' : ['1234','0.3']},
#                 '2235' : {'test_sample_02' : ['2235','202']},
#                 '1035' : {'test_sample_01' : ['1035','202','0.3'],
#                           'test_sample_02' : ['1035','0.3']},
#                 '1305' : {'test_sample_01' : ['1005','202','0.3']},
#                 '1350' : {'test_sample_01' : ['1350','202','0.3'],
#                           'test_sample_02' : ['1350','0.3']}}

#         self.TS.sort_data(data)

#         self.assertEqual(self.TS.sim_data, [['1035', '202', '0.3', '0.3'], 
#                                              ['1234', '202', '0.3', '0.3'], 
#                                              ['1305', '202', '0.3', '-'], 
#                                              ['1350', '202', '0.3', '0.3'], 
#                                              ['2235', '-', '-', '202']])

#     def test_remove_time_headers(self):
#         hdrs_list = ['A', 'B', 'time', 'TIME', 'C', 'D']
#         indices, clean_hdrs_list = self.TS.remove_time_headers(hdrs_list)
#         self.assertEqual(clean_hdrs_list, ['A', 'B','C', 'D'])

#     def test_remove_time_datapoints(self):
#         data = ['1', '2', '3', '4']
#         indices_to_remove = [0, 2]
#         clean_data = self.TS.remove_time_datapoints(data, indices_to_remove)
#         self.assertEqual(clean_data, ['2', '4'])

#         data = [['list1'], [], ['list3', 'list3'], []]
#         indices_to_remove = [0, 2]
#         clean_data = self.TS.remove_time_datapoints(data, indices_to_remove)
#         self.assertEqual(clean_data, [[],[]])

#     def test_get_vehicle_metadata(self):
#         return 

#     def test_get_sim_data(self):
#         return
        
# if __name__ == '__main__':
#     unittest.main()
