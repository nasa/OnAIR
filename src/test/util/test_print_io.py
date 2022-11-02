import pytest
from mock import MagicMock
import util.print_io


# bcolors tests
def test_bcolors_HEADER_is_expected_value():
  assert util.print_io.bcolors.HEADER == '\033[95m'

def test_bcolors_OKBLUE_is_expected_value():
  assert util.print_io.bcolors.OKBLUE == '\033[94m'

def test_bcolors_OKGREEN_is_expected_value():
  assert util.print_io.bcolors.OKGREEN == '\033[92m'
  
def test_bcolors_WARNING_is_expected_value():
  assert util.print_io.bcolors.WARNING == '\033[93m'

def test_bcolors_FAIL_is_expected_value():
  assert util.print_io.bcolors.FAIL == '\033[91m'

def test_bcolors_ENDC_is_expected_value():
  assert util.print_io.bcolors.ENDC == '\033[0m'

def test_bcolors_BOLD_is_expected_value():
  assert util.print_io.bcolors.BOLD == '\033[1m'
  
def test_bcolors_UNDERLINE_is_expected_value():
  assert util.print_io.bcolors.UNDERLINE == '\033[4m'


# Globals tests
def test_scolors_HEADER_is_set_to_bcolors_HEADER():
  assert util.print_io.scolors['HEADER'] == util.print_io.bcolors.HEADER

def test_scolors_OKBLUE_is_set_to_bcolors_OKBLUE():
  assert util.print_io.scolors['OKBLUE'] == util.print_io.bcolors.OKBLUE

def test_scolors_OKGREEN_is_set_to_bcolors_OKGREEN():
  assert util.print_io.scolors['OKGREEN'] == util.print_io.bcolors.OKGREEN
  
def test_scolors_WARNING_is_set_to_bcolors_WARNING():
  assert util.print_io.scolors['WARNING'] == util.print_io.bcolors.WARNING

def test_scolors_FAIL_is_set_to_bcolors_FAIL():
  assert util.print_io.scolors['FAIL'] == util.print_io.bcolors.FAIL

def test_scolors_ENDC_is_set_to_bcolors_ENDC():
  assert util.print_io.scolors['ENDC'] == util.print_io.bcolors.ENDC

def test_scolors_BOLD_is_set_to_bcolors_BOLD():
  assert util.print_io.scolors['BOLD'] == util.print_io.bcolors.BOLD
  
def test_scolors_UNDERLINE_is_set_to_bcolors_UNDERLINE():
  assert util.print_io.scolors['UNDERLINE'] == util.print_io.bcolors.UNDERLINE

def test_status_colors_GREEN_is_set_to_bcolors_OKGREEN():
  assert util.print_io.status_colors['GREEN'] == util.print_io.bcolors.OKGREEN

def test_status_colors_YELLOW_is_set_to_bcolors_WARNING():
  assert util.print_io.status_colors['YELLOW'] == util.print_io.bcolors.WARNING

def test_status_colors_RED_is_set_to_bcolors_FAIL():
  assert util.print_io.status_colors['RED'] == util.print_io.bcolors.FAIL

def test_status_colors_3_dashes_is_set_to_bcolors_OKBLUE():
  assert util.print_io.status_colors['---'] == util.print_io.bcolors.OKBLUE


# print_sim_header tests
def test_print_sim_header_prints_expected_strings(mocker):
  # Arrange
  expected_print = []
  expected_print.append(util.print_io.bcolors.HEADER + \
                        util.print_io.bcolors.BOLD +\
                        "\n***************************************************")
  expected_print.append("************    SIMULATION STARTED     ************")
  expected_print.append("***************************************************" + \
                        util.print_io.bcolors.ENDC)

  mocker.patch('util.print_io.print')

  # Act
  util.print_io.print_sim_header()

  # Assert
  for i in range(3):
    util.print_io.print.call_args_list[i].args == (expected_print[i], )


# print_sim_step tests
def test_print_sim_step_inserts_given_step_num_into_text(mocker):
  # Arrange
  arg_step_num = pytest.gen.randint(1, 100) # arbitrary from 1 to 100
  expected_print = util.print_io.bcolors.HEADER + \
                   util.print_io.bcolors.BOLD + \
                   f"\n--------------------- STEP {arg_step_num}" + \
                   " ---------------------\n" + \
                   util.print_io.bcolors.ENDC

  mocker.patch('util.print_io.print')

  # Act
  util.print_io.print_sim_step(arg_step_num)
  
  # Assert
  assert util.print_io.print.call_args_list[0].args == (expected_print, )


# print_seperator tests
def test_print_seperator_uses_bcolors_HEADER_as_default_color_value(mocker):
  # Arrange
  expected_color = util.print_io.bcolors.HEADER
  expected_print = expected_color + \
                   util.print_io.bcolors.BOLD + \
                   "\n------------------------------------------------\n" + \
                   util.print_io.bcolors.ENDC

  mocker.patch('util.print_io.print')

  # Act
  util.print_io.print_seperator()
  
  # Assert
  assert util.print_io.print.call_args_list[0].args == (expected_print, )

def test_print_seperator_prints_whatever_is_passed_in_as_color_at_start_of_line(mocker):
  # Arrange
  arg_color = MagicMock()

  expected_print = arg_color + \
                   util.print_io.bcolors.BOLD + \
                   "\n------------------------------------------------\n" + \
                   util.print_io.bcolors.ENDC

  mocker.patch('util.print_io.print')

  # Act
  util.print_io.print_seperator(arg_color)
  
  # Assert
  assert util.print_io.print.call_count == 1
  assert util.print_io.print.call_args_list[0].args == (expected_print, )


# update_header tests
def test_update_header_prints_message_with_bcolors_BOLD_at_start_when_no_clr_arg_given(mocker):
  # Arrange
  arg_msg = MagicMock()

  expected_clr = util.print_io.bcolors.BOLD
  expected_print = expected_clr + \
                   "--------- " + arg_msg + " update" + \
                   util.print_io.bcolors.ENDC

  mocker.patch('util.print_io.print')

  # Act
  util.print_io.update_header(arg_msg)

  # Assert
  assert util.print_io.print.call_count == 1
  assert util.print_io.print.call_args_list[0].args == (expected_print, )

def test_update_header_prints_message_starting_with_whatever_is_given_as_clr(mocker):
  # Arrange
  arg_msg = MagicMock()
  arg_clr = MagicMock()

  expected_print = arg_clr + \
                   "--------- " + arg_msg + " update" + \
                   util.print_io.bcolors.ENDC

  mocker.patch('util.print_io.print')

  # Act
  util.print_io.update_header(arg_msg, arg_clr)

  # Assert
  assert util.print_io.print.call_count == 1
  assert util.print_io.print.call_args_list[0].args == (expected_print, )


# print_msg tests
def test_print_msg_prints_message_starting_only_with_scolor_HEADER_when_no_clrs_arg_given(mocker):
    # Arrange
  arg_msg = MagicMock()
  
  expected_scolor = util.print_io.scolors['HEADER']
  expected_print = []
  expected_print.append(expected_scolor)
  expected_print.append("---- " + arg_msg + util.print_io.bcolors.ENDC)

  mocker.patch('util.print_io.print')

  # Act
  util.print_io.print_msg(arg_msg)

  # Assert
  assert util.print_io.print.call_count == 2
  for i in range(2):
   assert util.print_io.print.call_args_list[i].args == (expected_print[i], )

def test_print_msg_raises_KeyError_when_given_clrs_item_not_in_scolors(mocker):
    # Arrange
  arg_msg = MagicMock()
  arg_clrs = ['THIS-WILL-THROW-KEYERROR']

  mocker.patch('util.print_io.print')

  # Act
  with pytest.raises(KeyError) as e_info:
    util.print_io.print_msg(arg_msg, arg_clrs)

  # Assert
  assert str(e_info.value) == "'THIS-WILL-THROW-KEYERROR'"
  assert util.print_io.print.call_count == 0

def test_print_msg_prints_only_given_msg_when_given_clrs_is_empty(mocker):
    # Arrange
  arg_msg = MagicMock()
  arg_clrs = []

  mocker.patch('util.print_io.print')

  # Act
  util.print_io.print_msg(arg_msg, arg_clrs)

  # Assert
  assert util.print_io.print.call_count == 1
  assert util.print_io.print.call_args_list[0].args == ("---- " + arg_msg + util.print_io.bcolors.ENDC, )

def test_print_msg_prints_all_scolors_given_in_clrs(mocker):
    # Arrange
  arg_msg = MagicMock()
  arg_clrs = list(util.print_io.scolors.keys())
  pytest.gen.shuffle(arg_clrs) # change up the order to show it does not matter

  mocker.patch('util.print_io.print')

  # Act
  util.print_io.print_msg(arg_msg, arg_clrs)

  # Assert
  assert util.print_io.print.call_count == len(util.print_io.scolors.keys()) + 1
  for i in range(len(arg_clrs)):
    assert util.print_io.print.call_args_list[i].args == (util.print_io.scolors[arg_clrs[i]], )
  assert util.print_io.print.call_args_list[i + 1].args == ("---- " + arg_msg + util.print_io.bcolors.ENDC, )


#print_mission_status
def test_print_mission_status_only_prints_brain_formatted_status_when_data_not_given(mocker):
  # Arrange
  arg_brain = MagicMock()
  
  fake_mission_status = MagicMock()
  fake_status = MagicMock()

  expected_print = "INTERPRETED MISSION STATUS: " + str(fake_status)

  arg_brain.mission_status = fake_mission_status
  mocker.patch('util.print_io.format_status', return_value=fake_status)
  mocker.patch('util.print_io.print')

  # Act
  util.print_io.print_mission_status(arg_brain)
  
  # Assert
  assert util.print_io.format_status.call_count == 1
  assert util.print_io.format_status.call_args_list[0].args == (fake_mission_status,)
  assert util.print_io.print.call_count == 1
  assert util.print_io.print.call_args_list[0].args == (expected_print, )

def test_print_mission_status_only_prints_brain_formatted_status_when_data_given_is_None(mocker):
  # Arrange
  arg_brain = MagicMock()
  arg_data = None
  
  fake_mission_status = MagicMock()
  fake_status = MagicMock()

  expected_print = "INTERPRETED MISSION STATUS: " + str(fake_status)

  arg_brain.mission_status = fake_mission_status
  mocker.patch('util.print_io.format_status', return_value=fake_status)
  mocker.patch('util.print_io.print')

  # Act
  util.print_io.print_mission_status(arg_brain, arg_data)
  
  # Assert
  assert util.print_io.format_status.call_count == 1
  assert util.print_io.format_status.call_args_list[0].args == (fake_mission_status,)
  assert util.print_io.print.call_count == 1
  assert util.print_io.print.call_args_list[0].args == (expected_print, )

def test_print_mission_status_only_prints_brain_formatted_status_when_data_given_is_None(mocker):
  # Arrange
  arg_brain = MagicMock()
  arg_data = MagicMock()
  
  fake_mission_status = MagicMock()
  fake_status = MagicMock()

  expected_print = []
  expected_print.append("CURRENT TLM: " + str(arg_data))
  expected_print.append("INTERPRETED MISSION STATUS: " + str(fake_status))

  arg_brain.mission_status = fake_mission_status
  mocker.patch('util.print_io.format_status', return_value=fake_status)
  mocker.patch('util.print_io.print')

  # Act
  util.print_io.print_mission_status(arg_brain, arg_data)
  
  # Assert
  assert util.print_io.format_status.call_count == 1
  assert util.print_io.format_status.call_args_list[0].args == (fake_mission_status,)
  assert util.print_io.print.call_count == 2
  for i in range(util.print_io.print.call_count):
    assert util.print_io.print.call_args_list[i].args == (expected_print[i], )