import pytest
from mock import MagicMock
import src.util.print_io


# bcolors tests
def test_print_io_bcolors_HEADER_is_expected_value():
  assert src.util.print_io.bcolors.HEADER == '\033[95m'

def test_print_io_bcolors_OKBLUE_is_expected_value():
  assert src.util.print_io.bcolors.OKBLUE == '\033[94m'

def test_print_io_bcolors_OKGREEN_is_expected_value():
  assert src.util.print_io.bcolors.OKGREEN == '\033[92m'
  
def test_print_io_bcolors_WARNING_is_expected_value():
  assert src.util.print_io.bcolors.WARNING == '\033[93m'

def test_print_io_bcolors_FAIL_is_expected_value():
  assert src.util.print_io.bcolors.FAIL == '\033[91m'

def test_print_io_bcolors_ENDC_is_expected_value():
  assert src.util.print_io.bcolors.ENDC == '\033[0m'

def test_print_io_bcolors_BOLD_is_expected_value():
  assert src.util.print_io.bcolors.BOLD == '\033[1m'
  
def test_print_io_bcolors_UNDERLINE_is_expected_value():
  assert src.util.print_io.bcolors.UNDERLINE == '\033[4m'


# Globals tests
def test_print_io_scolors_HEADER_is_set_to_bcolors_HEADER():
  assert src.util.print_io.scolors['HEADER'] == src.util.print_io.bcolors.HEADER

def test_print_io_scolors_OKBLUE_is_set_to_bcolors_OKBLUE():
  assert src.util.print_io.scolors['OKBLUE'] == src.util.print_io.bcolors.OKBLUE

def test_print_io_scolors_OKGREEN_is_set_to_bcolors_OKGREEN():
  assert src.util.print_io.scolors['OKGREEN'] == src.util.print_io.bcolors.OKGREEN
  
def test_print_io_scolors_WARNING_is_set_to_bcolors_WARNING():
  assert src.util.print_io.scolors['WARNING'] == src.util.print_io.bcolors.WARNING

def test_print_io_scolors_FAIL_is_set_to_bcolors_FAIL():
  assert src.util.print_io.scolors['FAIL'] == src.util.print_io.bcolors.FAIL

def test_print_io_scolors_ENDC_is_set_to_bcolors_ENDC():
  assert src.util.print_io.scolors['ENDC'] == src.util.print_io.bcolors.ENDC

def test_print_io_scolors_BOLD_is_set_to_bcolors_BOLD():
  assert src.util.print_io.scolors['BOLD'] == src.util.print_io.bcolors.BOLD
  
def test_print_io_scolors_UNDERLINE_is_set_to_bcolors_UNDERLINE():
  assert src.util.print_io.scolors['UNDERLINE'] == src.util.print_io.bcolors.UNDERLINE

def test_print_io_status_colors_GREEN_is_set_to_bcolors_OKGREEN():
  assert src.util.print_io.status_colors['GREEN'] == src.util.print_io.bcolors.OKGREEN

def test_print_io_status_colors_YELLOW_is_set_to_bcolors_WARNING():
  assert src.util.print_io.status_colors['YELLOW'] == src.util.print_io.bcolors.WARNING

def test_print_io_status_colors_RED_is_set_to_bcolors_FAIL():
  assert src.util.print_io.status_colors['RED'] == src.util.print_io.bcolors.FAIL

def test_print_io_status_colors_3_dashes_is_set_to_bcolors_OKBLUE():
  assert src.util.print_io.status_colors['---'] == src.util.print_io.bcolors.OKBLUE


# print_sim_header tests
def test_print_io_print_sim_header_prints_expected_strings(mocker):
  # Arrange
  expected_print = []
  expected_print.append(src.util.print_io.bcolors.HEADER + \
                        src.util.print_io.bcolors.BOLD +\
                        "\n***************************************************")
  expected_print.append("************    SIMULATION STARTED     ************")
  expected_print.append("***************************************************" + \
                        src.util.print_io.bcolors.ENDC)

  mocker.patch('src.util.print_io.print')

  # Act
  src.util.print_io.print_sim_header()

  # Assert
  for i in range(3):
    src.util.print_io.print.call_args_list[i].args == (expected_print[i], )


# print_sim_step tests
def test_print_io_print_sim_step_inserts_given_step_num_into_text(mocker):
  # Arrange
  arg_step_num = pytest.gen.randint(1, 100) # arbitrary from 1 to 100
  expected_print = src.util.print_io.bcolors.HEADER + \
                   src.util.print_io.bcolors.BOLD + \
                   f"\n--------------------- STEP {arg_step_num}" + \
                   " ---------------------\n" + \
                   src.util.print_io.bcolors.ENDC

  mocker.patch('src.util.print_io.print')

  # Act
  src.util.print_io.print_sim_step(arg_step_num)
  
  # Assert
  assert src.util.print_io.print.call_args_list[0].args == (expected_print, )


# print_separator tests
def test_print_io_print_separator_uses_bcolors_HEADER_as_default_color_value(mocker):
  # Arrange
  expected_color = src.util.print_io.bcolors.HEADER
  expected_print = expected_color + \
                   src.util.print_io.bcolors.BOLD + \
                   "\n------------------------------------------------\n" + \
                   src.util.print_io.bcolors.ENDC

  mocker.patch('src.util.print_io.print')

  # Act
  src.util.print_io.print_separator()
  
  # Assert
  assert src.util.print_io.print.call_args_list[0].args == (expected_print, )

def test_print_io_print_separator_prints_whatever_is_passed_in_as_color_at_start_of_line(mocker):
  # Arrange
  arg_color = MagicMock()

  expected_print = arg_color + \
                   src.util.print_io.bcolors.BOLD + \
                   "\n------------------------------------------------\n" + \
                   src.util.print_io.bcolors.ENDC

  mocker.patch('src.util.print_io.print')

  # Act
  src.util.print_io.print_separator(arg_color)
  
  # Assert
  assert src.util.print_io.print.call_count == 1
  assert src.util.print_io.print.call_args_list[0].args == (expected_print, )


# update_header tests
def test_print_io_update_header_prints_message_with_bcolors_BOLD_at_start_when_no_clr_arg_given(mocker):
  # Arrange
  arg_msg = MagicMock()

  expected_clr = src.util.print_io.bcolors.BOLD
  expected_print = expected_clr + \
                   "--------- " + arg_msg + " update" + \
                   src.util.print_io.bcolors.ENDC

  mocker.patch('src.util.print_io.print')

  # Act
  src.util.print_io.update_header(arg_msg)

  # Assert
  assert src.util.print_io.print.call_count == 1
  assert src.util.print_io.print.call_args_list[0].args == (expected_print, )

def test_print_io_update_header_prints_message_starting_with_whatever_is_given_as_clr(mocker):
  # Arrange
  arg_msg = MagicMock()
  arg_clr = MagicMock()

  expected_print = arg_clr + \
                   "--------- " + arg_msg + " update" + \
                   src.util.print_io.bcolors.ENDC

  mocker.patch('src.util.print_io.print')

  # Act
  src.util.print_io.update_header(arg_msg, arg_clr)

  # Assert
  assert src.util.print_io.print.call_count == 1
  assert src.util.print_io.print.call_args_list[0].args == (expected_print, )


# print_msg tests
def test_print_io_print_msg_prints_message_starting_only_with_scolor_HEADER_when_no_clrs_arg_given(mocker):
    # Arrange
  arg_msg = MagicMock()
  
  expected_scolor = src.util.print_io.scolors['HEADER']
  expected_print = []
  expected_print.append(expected_scolor)
  expected_print.append("---- " + arg_msg + src.util.print_io.bcolors.ENDC)

  mocker.patch('src.util.print_io.print')

  # Act
  src.util.print_io.print_msg(arg_msg)

  # Assert
  assert src.util.print_io.print.call_count == 2
  for i in range(2):
   assert src.util.print_io.print.call_args_list[i].args == (expected_print[i], )

def test_print_io_print_msg_raises_KeyError_when_given_clrs_item_not_in_scolors(mocker):
    # Arrange
  arg_msg = MagicMock()
  arg_clrs = ['THIS-WILL-THROW-KEYERROR']

  mocker.patch('src.util.print_io.print')

  # Act
  with pytest.raises(KeyError) as e_info:
    src.util.print_io.print_msg(arg_msg, arg_clrs)

  # Assert
  assert str(e_info.value) == "'THIS-WILL-THROW-KEYERROR'"
  assert src.util.print_io.print.call_count == 0

def test_print_io_print_msg_prints_only_given_msg_when_given_clrs_is_empty(mocker):
    # Arrange
  arg_msg = MagicMock()
  arg_clrs = []

  mocker.patch('src.util.print_io.print')

  # Act
  src.util.print_io.print_msg(arg_msg, arg_clrs)

  # Assert
  assert src.util.print_io.print.call_count == 1
  assert src.util.print_io.print.call_args_list[0].args == ("---- " + arg_msg + src.util.print_io.bcolors.ENDC, )

def test_print_io_print_msg_prints_all_scolors_given_in_clrs(mocker):
    # Arrange
  arg_msg = MagicMock()
  arg_clrs = list(src.util.print_io.scolors.keys())
  pytest.gen.shuffle(arg_clrs) # change up the order to show it does not matter

  mocker.patch('src.util.print_io.print')

  # Act
  src.util.print_io.print_msg(arg_msg, arg_clrs)

  # Assert
  assert src.util.print_io.print.call_count == len(src.util.print_io.scolors.keys()) + 1
  for i in range(len(arg_clrs)):
    assert src.util.print_io.print.call_args_list[i].args == (src.util.print_io.scolors[arg_clrs[i]], )
  assert src.util.print_io.print.call_args_list[i + 1].args == ("---- " + arg_msg + src.util.print_io.bcolors.ENDC, )


#print_mission_status
def test_print_io_print_mission_status_only_prints_agent_formatted_status_when_data_not_given(mocker):
  # Arrange
  arg_agent = MagicMock()
  
  fake_mission_status = MagicMock()
  fake_status = MagicMock()

  expected_print = "INTERPRETED SYSTEM STATUS: " + str(fake_status)

  arg_agent.mission_status = fake_mission_status
  mocker.patch('src.util.print_io.format_status', return_value=fake_status)
  mocker.patch('src.util.print_io.print')

  # Act
  src.util.print_io.print_system_status(arg_agent)
  
  # Assert
  assert src.util.print_io.format_status.call_count == 1
  assert src.util.print_io.format_status.call_args_list[0].args == (fake_mission_status,)
  assert src.util.print_io.print.call_count == 1
  assert src.util.print_io.print.call_args_list[0].args == (expected_print, )

def test_print_io_print_mission_status_only_prints_agent_formatted_status_when_data_given_is_None(mocker):
  # Arrange
  arg_agent = MagicMock()
  arg_data = None
  
  fake_mission_status = MagicMock()
  fake_status = MagicMock()

  expected_print = "INTERPRETED SYSTEM STATUS: " + str(fake_status)

  arg_agent.mission_status = fake_mission_status
  mocker.patch('src.util.print_io.format_status', return_value=fake_status)
  mocker.patch('src.util.print_io.print')

  # Act
  src.util.print_io.print_system_status(arg_agent, arg_data)
  
  # Assert
  assert src.util.print_io.format_status.call_count == 1
  assert src.util.print_io.format_status.call_args_list[0].args == (fake_mission_status,)
  assert src.util.print_io.print.call_count == 1
  assert src.util.print_io.print.call_args_list[0].args == (expected_print, )

def test_print_io_print_mission_status_only_prints_agent_formatted_status_when_data_given_is_None(mocker):
  # Arrange
  arg_agent = MagicMock()
  arg_data = MagicMock()
  
  fake_mission_status = MagicMock()
  fake_status = MagicMock()

  expected_print = []
  expected_print.append("CURRENT DATA: " + str(arg_data))
  expected_print.append("INTERPRETED SYSTEM STATUS: " + str(fake_status))

  arg_agent.mission_status = fake_mission_status
  mocker.patch('src.util.print_io.format_status', return_value=fake_status)
  mocker.patch('src.util.print_io.print')

  # Act
  src.util.print_io.print_system_status(arg_agent, arg_data)
  
  # Assert
  assert src.util.print_io.format_status.call_count == 1
  assert src.util.print_io.format_status.call_args_list[0].args == (fake_mission_status,)
  assert src.util.print_io.print.call_count == 2
  for i in range(src.util.print_io.print.call_count):
    assert src.util.print_io.print.call_args_list[i].args == (expected_print[i], )


# print_diagnosis tests
def test_print_io_print_diagnosis_only_prints_separators_and_headers_when_status_list_and_activations_are_empty_tree_traversal_unused(mocker):
  # Arrange
  arg_diagnosis = MagicMock()

  arg_diagnosis.configure_mock(**{'get_status_list.return_value': []})
  arg_diagnosis.configure_mock(**{'current_activations.return_value': []})

  mocker.patch('src.util.print_io.print_separator')
  mocker.patch('src.util.print_io.print')

  # Act
  src.util.print_io.print_diagnosis(arg_diagnosis)

  # Assert
  assert src.util.print_io.print_separator.call_count == 2
  assert src.util.print_io.print.call_count == 2
  assert src.util.print_io.print.call_args_list[0].args == (src.util.print_io.bcolors.HEADER + src.util.print_io.bcolors.BOLD + "DIAGNOSIS INFO: \n" + src.util.print_io.bcolors.ENDC, )
  assert src.util.print_io.print.call_args_list[1].args == (src.util.print_io.bcolors.HEADER + src.util.print_io.bcolors.BOLD + "\nCURRENT ACTIVATIONS: \n" + src.util.print_io.bcolors.ENDC, )

def test_print_io_print_diagnosis_prints_separators_headers_status_and_activations_when_status_list_and_activations_have_items_tree_traversal_unused(mocker):
  # Arrange
  arg_diagnosis = MagicMock()

  num_status = pytest.gen.randint(1, 10) # arbitrary from 1 to 10
  fake_status = []
  fake_format = MagicMock()
  num_activations = pytest.gen.randint(1, 10) # arbitrary from 1 to 10
  fake_activations = []
  fake_str = MagicMock()

  for i in range(num_status):
    fake_status.append([MagicMock(), MagicMock()])

  for i in range(num_activations):
    fake_activations.append(MagicMock())

  arg_diagnosis.configure_mock(**{'get_status_list.return_value': fake_status})
  arg_diagnosis.current_activations = fake_activations

  mocker.patch('src.util.print_io.print_separator')
  mocker.patch('src.util.print_io.print')
  mocker.patch('src.util.print_io.format_status', return_value=fake_format)
  mocker.patch('src.util.print_io.str', return_value=fake_str)

  # Act
  src.util.print_io.print_diagnosis(arg_diagnosis)

  # Assert
  assert src.util.print_io.print_separator.call_count == 2
  assert src.util.print_io.print.call_count == 2 + num_status + num_activations
  assert src.util.print_io.print.call_args_list[0].args == (src.util.print_io.bcolors.HEADER + src.util.print_io.bcolors.BOLD + "DIAGNOSIS INFO: \n" + src.util.print_io.bcolors.ENDC, )
  for i in range(num_status):
    assert src.util.print_io.print.call_args_list[1 + i].args == (fake_status[i][0] + ': ' + fake_format, )
    assert src.util.print_io.format_status.call_args_list[i].args == (fake_status[i][1], )
  assert src.util.print_io.print.call_args_list[1 + num_status].args == (src.util.print_io.bcolors.HEADER + src.util.print_io.bcolors.BOLD + "\nCURRENT ACTIVATIONS: \n" + src.util.print_io.bcolors.ENDC, )
  for i in range(num_activations):
    assert src.util.print_io.print.call_args_list[2 + num_status + i].args == ('---' + fake_str, )
    assert src.util.print_io.str.call_args_list[i].args == (fake_activations[i], )


# subsystem_status_str tests
def test_print_io_subsystem_status_str_returns_expected_string_when_stat_exists_as_key_in_status_colors(mocker):
  # Arrange
  arg_ss = MagicMock()

  fake_type = MagicMock()
  fake_stat = pytest.gen.choice(list(src.util.print_io.status_colors.keys()))
  fake_uncertainty = MagicMock()
  fake_str = MagicMock()

  expected_s = src.util.print_io.bcolors.BOLD + '[' + fake_str + '] : ' + src.util.print_io.bcolors.ENDC
  expected_s = expected_s + '\n' + src.util.print_io.status_colors[fake_stat] + ' ---- ' + fake_str + src.util.print_io.bcolors.ENDC + ' (' + fake_str + ')'
  expected_s = expected_s + '\n'

  arg_ss.type = fake_type
  arg_ss.configure_mock(**{'get_status.return_value':fake_stat})
  arg_ss.uncertainty = fake_uncertainty

  mocker.patch('src.util.print_io.str', return_value=fake_str)

  # Act
  result = src.util.print_io.subsystem_status_str(arg_ss)

  # Assert
  assert src.util.print_io.str.call_count == 3
  assert src.util.print_io.str.call_args_list[0].args == (fake_type, )
  assert src.util.print_io.str.call_args_list[1].args == (fake_stat, )
  assert src.util.print_io.str.call_args_list[2].args == (fake_uncertainty, )
  assert result == expected_s
  

# subsystem_str tests
def test_print_io_subsystem_str_returns_string_without_any_data_when_headers_tests_and_test_data_empty(mocker):
  # Arrange
  arg_ss = MagicMock()

  arg_ss.type = str(MagicMock())
  arg_ss.headers = []
  arg_ss.tests = []
  arg_ss.test_data = []

  expected_result = src.util.print_io.bcolors.BOLD + arg_ss.type + '\n' + src.util.print_io.bcolors.ENDC
  expected_result = expected_result + '--[headers] \n--[tests] \n--[test data] '

  # Act
  result = src.util.print_io.subsystem_str(arg_ss)

  # Assert
  assert result == expected_result

def test_print_io_subsystem_str_returns_string_all_data_when_headers_tests_and_test_data_occupied(mocker):
  # Arrange
  arg_ss = MagicMock()

  arg_ss.type = str(MagicMock())
  num_headers = pytest.gen.randint(1, 10) # arbitrary from 1 to 10
  arg_ss.headers = []
  num_tests = pytest.gen.randint(1, 10) # arbitrary from 1 to 10
  arg_ss.tests = []
  num_test_data = pytest.gen.randint(1, 10) # arbitrary from 1 to 10
  arg_ss.test_data = []

  expected_result = src.util.print_io.bcolors.BOLD + arg_ss.type + '\n' + src.util.print_io.bcolors.ENDC
  expected_result = expected_result + '--[headers] '
  for i in range(num_headers):
    arg_ss.headers.append(MagicMock())
    expected_result = expected_result + '\n---' + str(arg_ss.headers[i])
  expected_result = expected_result + '\n--[tests] '
  for i in range(num_tests):
    arg_ss.tests.append(MagicMock())
    expected_result = expected_result + '\n---' + str(arg_ss.tests[i])
  expected_result = expected_result + '\n--[test data] '
  for i in range(num_test_data):
    arg_ss.test_data.append(MagicMock())
    expected_result = expected_result + '\n---' + str(arg_ss.test_data[i])

  # Act
  result = src.util.print_io.subsystem_str(arg_ss)

  # Assert
  assert result == expected_result


# headers_string tests
def test_print_io_format_status_returns_empty_string_when_headers_is_vacant():
  # Arrange
  arg_headers = []

  # Act
  result = src.util.print_io.headers_string(arg_headers)

  # Assert
  assert result == str()

def test_print_io_format_status_returns_all_headers_in_formatted_string_when_occupied():
  # Arrange
  num_headers = pytest.gen.randint(1, 10) # arbitrary from 1 to 10
  arg_headers = []
  
  expected_result = ''

  for i in range(num_headers):
    arg_headers.append(str(MagicMock()))
    expected_result = expected_result + '\n  -- ' + arg_headers[i]
  
  # Act
  result = src.util.print_io.headers_string(arg_headers)

  # Assert
  assert result == expected_result


# format_status tests

def test_print_io_format_status_raises_KeyError_when_stat_is_string_and_not_in_status_color_keys():
  # Arrange
  arg_stat = str(MagicMock())

  # Act
  with pytest.raises(KeyError) as e_info:
    result = src.util.print_io.format_status(arg_stat)

  # Assert
  assert str(e_info.value) == '"' + arg_stat + '"'

def test_print_io_format_status_returns_stat_in_its_status_color_when_stat_is_string_and_a_key():
  # Arrange
  arg_stat = pytest.gen.choice(list(src.util.print_io.status_colors.keys()))

  expected_result = src.util.print_io.status_colors[arg_stat] + arg_stat + src.util.print_io.scolors['ENDC']

  # Act
  result = src.util.print_io.format_status(arg_stat)

  # Assert
  assert result == expected_result

def test_print_io_format_status_returns_only_a_right_parenthesis_in_string_when_stat_is_an_empty_list():
  # Arrange
  arg_stat = []

  expected_result = ')'

  # Act
  result = src.util.print_io.format_status(arg_stat)

  # Assert
  assert result == expected_result

def test_print_io_format_status_returns_all_status_in_stat_formatted_into_string_when_stat_is_a_list_of_status(mocker):
  # Arrange
  num_stat = pytest.gen.randint(1, 10) # arbitrary from 1 to 10
  arg_stat = []

  expected_result = '('
  for i in range(num_stat):
    arg_stat.append(pytest.gen.choice(list(src.util.print_io.status_colors.keys())))
    expected_result += src.util.print_io.status_colors[arg_stat[i]] + arg_stat[i] + src.util.print_io.scolors['ENDC']
    if i != (num_stat - 1):
      expected_result += ', '
  expected_result += ')'

  # Act
  result = src.util.print_io.format_status(arg_stat)

  # Assert
  assert result == expected_result