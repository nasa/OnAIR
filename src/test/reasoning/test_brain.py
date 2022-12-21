""" Test Brain Functionality """
import pytest
from mock import MagicMock
import src.reasoning.brain as brain
from src.reasoning.brain import Brain


# __init__ tests

# reason tests
def test_reason_updates_spacecraft_rep_with_given_frame_and_sets_new_spacecraft_mission_status_and_updates_learning_systems_with_given_frame_and_new_mission_status(mocker):
    # Arrange
    arg_frame = MagicMock()

    fake_mission_status = MagicMock()

    cut = Brain.__new__(Brain)
    cut.spacecraft_rep = MagicMock()
    cut.learning_systems = MagicMock()
    
    mocker.patch.object(cut.spacecraft_rep, 'update')
    mocker.patch.object(cut.spacecraft_rep, 'get_status', return_value=fake_mission_status)
    mocker.patch.object(cut.learning_systems, 'update')

    # Act
    cut.reason(arg_frame)

    # Assert
    assert cut.spacecraft_rep.update.call_count == 1  
    assert cut.spacecraft_rep.update.call_args_list[0].args == (arg_frame, ) 
    assert cut.spacecraft_rep.get_status.call_count == 1  
    assert cut.spacecraft_rep.get_status.call_args_list[0].args == () 
    assert cut.learning_systems.update.call_count == 1  
    assert cut.learning_systems.update.call_args_list[0].args == (arg_frame, fake_mission_status) 
     
# diagnose tests
def test_diagnose_returns_None():
    # Arrange
    arg_time_step = MagicMock()

    cut = Brain.__new__(Brain)

    # Act
    result = cut.diagnose(arg_time_step)

    # Assert
    assert result == None

# class TestBrain(unittest.TestCase):

#     def setUp(self):
#         self.test_path = os.path.dirname(os.path.abspath(__file__))
#         SC = Spacecraft(['TIME', 'A', 'B'], [[['SYNC', 'TIME']], [['NOOP']], [['NOOP']]])
#         self.B = Brain(SC)

#     def test_init_nonempty_brain(self):
#         self.assertEqual(type(self.B.spacecraft_rep), Spacecraft)
#         self.assertEqual(self.B.mission_status, '---')
#         self.assertEqual(self.B.bayesian_status, ('---', -1.0))

#     def test_reason(self):
#         frame = [1, 1, 1]
#         self.B.reason(frame)
#         self.assertEqual(self.B.spacecraft_rep.get_current_data(), [1,1,1])
#         self.assertEqual(self.B.mission_status, 'GREEN')

#     def test_diagnose(self):
#         return
        

# if __name__ == '__main__':
#     unittest.main()
