""" Test Generic Component Core Functionality """
import pytest
from mock import MagicMock

import src.data_driven_components.generic_component.core as core
from src.data_driven_components.generic_component.core import AIPlugIn

# __init__ tests
def test_AIPlugIn__init__asserts_when_given__headers_len_is_less_than_0():
    # Arrange
    arg__name = MagicMock()
    arg__headers = []

    cut = AIPlugIn.__new__(AIPlugIn)

    # Act
    with pytest.raises(AssertionError) as e_info:
        cut.__init__(arg__name, arg__headers)

    # Assert
    assert e_info.match('')

def test_AIPlugIn__init__sets_instance_values_to_given_args_when_when_given__headers_len_is_greater_than_0(mocker):
    # Arrange
    arg__name = MagicMock()
    arg__headers = MagicMock()

    cut = AIPlugIn.__new__(AIPlugIn)

    mocker.patch('src.data_driven_components.generic_component.core.len', return_value=pytest.gen.randint(1, 200)) # arbitrary, from 1 to 200 (but > 0)

    # Act
    cut.__init__(arg__name, arg__headers)

    # Assert
    assert core.len.call_count == 1
    assert core.len.call_args_list[0].args == (arg__headers,)
    assert cut.component_name == arg__name
    assert cut.headers == arg__headers

# apriori_training tests
def test_AIPlugIn_apriori_training_returns_None():
    # Arrange
    arg_batch_data = MagicMock()

    cut = AIPlugIn.__new__(AIPlugIn)

    # Act
    result = cut.apriori_training(arg_batch_data)

    # Assert
    assert result == None 

# update tests
def test_AIPlugIn_update_asserts_when_len_of_given_frame_is_not_eq_len_of_instance_headers_and_given_frame_has_at_least_1_item(mocker):
    # Arrange
    arg_frame = []
    
    num_fake_frame = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10 (0 has own test)
    for i in range(num_fake_frame):
        arg_frame.append(MagicMock())
    fake_headers = MagicMock()
    len_fake_headers = num_fake_frame + (1 if pytest.gen.randint(0, 1) else -1) # arbitrary, 50/50 add or subtract 1

    cut = AIPlugIn.__new__(AIPlugIn)
    cut.headers = fake_headers

    mocker.patch('src.data_driven_components.generic_component.core.len', side_effect=[num_fake_frame, len_fake_headers])

    # Act
    with pytest.raises(AssertionError) as e_info:
        cut.update(arg_frame)

    # Assert
    assert e_info.match('')
    assert core.len.call_count == 2
    assert core.len.call_args_list[0].args == (arg_frame, )
    assert core.len.call_args_list[1].args == (fake_headers, )
    
def test_AIPlugIn_update_asserts_when_len_of_given_frame_is_not_eq_len_of_instance_headers_and_given_frame_has_no_items(mocker):
    # Arrange
    arg_frame = []

    fake_headers = MagicMock()
    len_fake_headers = pytest.gen.randint(1, 200) # arbitrary, from 1 to 200 for > 0

    cut = AIPlugIn.__new__(AIPlugIn)
    cut.headers = fake_headers

    mocker.patch('src.data_driven_components.generic_component.core.len', side_effect=[0, len_fake_headers])

    # Act
    with pytest.raises(AssertionError) as e_info:
        cut.update(arg_frame)

    # Assert
    assert e_info.match('')
    assert core.len.call_count == 2
    assert core.len.call_args_list[0].args == (arg_frame, )
    assert core.len.call_args_list[1].args == (fake_headers, )
    
def test_AIPlugIn_update_does_not_assert_when_len_of_given_frame_eq_len_of_instance_headers(mocker):
    # Arrange
    arg_frame = []
    
    num_fake_frame = pytest.gen.randint(0, 10) # arbitrary, from 0 to 10
    for i in range(num_fake_frame):
        arg_frame.append(MagicMock())
    fake_headers = MagicMock()
    len_fake_headers = num_fake_frame

    cut = AIPlugIn.__new__(AIPlugIn)
    cut.headers = fake_headers

    mocker.patch('src.data_driven_components.generic_component.core.len', side_effect=[num_fake_frame, len_fake_headers])

    # Act
    cut.update(arg_frame)

    # Assert
    assert core.len.call_count == 2
    assert core.len.call_args_list[0].args == (arg_frame, )
    assert core.len.call_args_list[1].args == (fake_headers, )
    
def test_AIPlugIn_default_frame_is_empty_list(mocker):
    # Arrange
    fake_headers = []

    cut = AIPlugIn.__new__(AIPlugIn)
    cut.headers = fake_headers

    mocker.patch('src.data_driven_components.generic_component.core.len')

    # Act
    cut.update()

    # Assert
    assert core.len.call_count == 2
    assert core.len.call_args_list[0].args == ([], )
    assert core.len.call_args_list[1].args == (fake_headers, )

# render_diagnosis tests
def test_AIPlugIn_render_diagnosis_returns_empty_list():
    # Arrange
    cut = AIPlugIn.__new__(AIPlugIn)

    # Act
    result = cut.render_diagnosis()

    # Assert
    assert result == []

# class TestCore(unittest.TestCase):

#     def setUp(self):
#         construct = AIPlugIn('test_component', ['test_A','test_B','test_C'])

#     def test_init_empty_headers(self):
#         self.assertRaises(AssertionError, AIPlugIn, 'test', [])

#     def test_init_non_empty_headers(self):
#         plugin = AIPlugIn('test_component', ['A'])
#         self.assertEqual(plugin.headers, ['A'])

#         plugin = AIPlugIn('test_component', ['A', 'B'])
#         self.assertEqual(plugin.headers, ['A', 'B'])

#     def apriori_training_empty_batch(self):
#         construct.apriori_training([])

#     def apriori_training_non_empty_batch(self):
#         construct.apriori_training([[1.0,2.0]])

#     def update_empty_frame(self):
#         self.assertRaises(AssertionError, construct.update([]))
        
#     def update_non_empty_frame(self):
#         construct.update([1.0,2.0,3.0])
#         # do we not do an assert here?

#     def render_diagnosis(self):
#         diagnosis = construct.render_diagnosis()
#         self.assertIsInstance(diagnosis, list)
#         if len(diagnosis>0):
#             for tlm in diagnosis:
#                 self.assertIn(tlm, construct.headers)
#         else:
#             self.assertEqual(diagnosis, [])

# if __name__ == '__main__':
#     unittest.main()
