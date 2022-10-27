from mock import MagicMock
import util.sim_io

def test_render_viz_does_only_stattest_render_viz_does_status_sensor_and_diagnosis_reports_when_diagnosis_is_givenus_and_sensor_reports_when_diagnosis_is_not_given(mocker):
  # Arrange
  SAVE_PATH = 'RAISR_VIZ_SAVE_PATH'
  arg_status_data = MagicMock()
  arg_sensor_data = MagicMock()
  arg_sim_name = MagicMock()

  fake_system_filename = MagicMock()
  fake_full_path = MagicMock()
  fake_iterator = MagicMock()
  fake_file = MagicMock()
  fake_file.configure_mock(**{'__enter__.return_value': fake_iterator})

  expected_status_report = {}
  expected_status_report['filename'] = arg_sim_name
  expected_status_report['data'] = arg_status_data
  expected_sensor_status_report = {}
  expected_sensor_status_report['name'] = 'MISSION'
  expected_sensor_status_report['children'] = arg_sensor_data

  mocker.patch('util.sim_io.os.environ.get', return_value=fake_system_filename)
  mocker.patch('util.sim_io.os.path.join', return_value=fake_full_path)
  mocker.patch('builtins.open', return_value=fake_file)
  mocker.patch('util.sim_io.json.dump')

  # Act
  util.sim_io.render_viz(arg_status_data, arg_sensor_data, arg_sim_name)

  # Assert
  assert open.call_count == 2
  assert util.sim_io.os.environ.get.call_args_list[0].args == (SAVE_PATH,)
  assert util.sim_io.os.path.join.call_args_list[0].args == (fake_system_filename, 'system.json')
  assert open.call_args_list[0].args == (fake_full_path, 'w')
  assert util.sim_io.json.dump.call_args_list[0].args == (expected_status_report, fake_iterator)
  assert util.sim_io.os.environ.get.call_args_list[1].args == (SAVE_PATH,)
  assert util.sim_io.os.path.join.call_args_list[1].args == (fake_system_filename, 'faults.json')
  assert open.call_args_list[1].args == (fake_full_path, 'w')
  assert util.sim_io.json.dump.call_args_list[1].args == (expected_sensor_status_report, fake_iterator)
  
def test_render_viz_does_only_status_and_sensor_reports_when_diagnosis_is_given_as_None(mocker):
  # Arrange
  SAVE_PATH = 'RAISR_VIZ_SAVE_PATH'
  arg_status_data = MagicMock()
  arg_sensor_data = MagicMock()
  arg_sim_name = MagicMock()
  arg_diagnosis = None

  fake_system_filename = MagicMock()
  fake_full_path = MagicMock()
  fake_iterator = MagicMock()
  fake_file = MagicMock()
  fake_file.configure_mock(**{'__enter__.return_value': fake_iterator})

  expected_status_report = {}
  expected_status_report['filename'] = arg_sim_name
  expected_status_report['data'] = arg_status_data
  expected_sensor_status_report = {}
  expected_sensor_status_report['name'] = 'MISSION'
  expected_sensor_status_report['children'] = arg_sensor_data

  mocker.patch('util.sim_io.os.environ.get', return_value=fake_system_filename)
  mocker.patch('util.sim_io.os.path.join', return_value=fake_full_path)
  mocker.patch('builtins.open', return_value=fake_file)
  mocker.patch('util.sim_io.json.dump')

  # Act
  util.sim_io.render_viz(arg_status_data, arg_sensor_data, arg_sim_name, arg_diagnosis)

  # Assert
  assert open.call_count == 2
  assert util.sim_io.os.environ.get.call_args_list[0].args == (SAVE_PATH,)
  assert util.sim_io.os.path.join.call_args_list[0].args == (fake_system_filename, 'system.json')
  assert open.call_args_list[0].args == (fake_full_path, 'w')
  assert util.sim_io.json.dump.call_args_list[0].args == (expected_status_report, fake_iterator)
  assert util.sim_io.os.environ.get.call_args_list[1].args == (SAVE_PATH,)
  assert util.sim_io.os.path.join.call_args_list[1].args == (fake_system_filename, 'faults.json')
  assert open.call_args_list[1].args == (fake_full_path, 'w')
  assert util.sim_io.json.dump.call_args_list[1].args == (expected_sensor_status_report, fake_iterator)
  
def test_render_viz_does_status_sensor_and_diagnosis_reports_when_diagnosis_is_given(mocker):
  # Arrange
  SAVE_PATH = 'RAISR_VIZ_SAVE_PATH'
  arg_status_data = MagicMock()
  arg_sensor_data = MagicMock()
  arg_sim_name = MagicMock()
  arg_diagnosis = MagicMock()

  fake_system_filename = MagicMock()
  fake_full_path = MagicMock()
  fake_iterator = MagicMock()
  fake_file = MagicMock()
  fake_file.configure_mock(**{'__enter__.return_value': fake_iterator})
  fake_results = MagicMock()

  expected_status_report = {}
  expected_status_report['filename'] = arg_sim_name
  expected_status_report['data'] = arg_status_data
  expected_sensor_status_report = {}
  expected_sensor_status_report['name'] = 'MISSION'
  expected_sensor_status_report['children'] = arg_sensor_data

  mocker.patch('util.sim_io.os.environ.get', return_value=fake_system_filename)
  mocker.patch('util.sim_io.os.path.join', return_value=fake_full_path)
  mocker.patch('builtins.open', return_value=fake_file)
  mocker.patch('util.sim_io.json.dump')
  arg_diagnosis.configure_mock(**{'get_diagnosis_viz_json.return_value': fake_results})

  # Act
  util.sim_io.render_viz(arg_status_data, arg_sensor_data, arg_sim_name, arg_diagnosis)

  # Assert
  assert open.call_count == 3
  assert util.sim_io.os.environ.get.call_args_list[0].args == (SAVE_PATH,)
  assert util.sim_io.os.path.join.call_args_list[0].args == (fake_system_filename, 'system.json')
  assert open.call_args_list[0].args == (fake_full_path, 'w')
  assert util.sim_io.json.dump.call_args_list[0].args == (expected_status_report, fake_iterator)
  assert util.sim_io.os.environ.get.call_args_list[1].args == (SAVE_PATH,)
  assert util.sim_io.os.path.join.call_args_list[1].args == (fake_system_filename, 'faults.json')
  assert open.call_args_list[1].args == (fake_full_path, 'w')
  assert util.sim_io.json.dump.call_args_list[1].args == (expected_sensor_status_report, fake_iterator)
  arg_diagnosis.get_diagnosis_viz_json.called_once()
  assert util.sim_io.os.environ.get.call_args_list[2].args == (SAVE_PATH,)
  assert util.sim_io.os.path.join.call_args_list[2].args == (fake_system_filename, 'results.json')
  assert open.call_args_list[2].args == (fake_full_path, 'w')
  assert util.sim_io.json.dump.call_args_list[2].args == (fake_results, fake_iterator)

def test_print_dots_uses_mod_10_plus_one_dots_when_ts_mod_20_is_less_than_10(mocker):
  # Arrange
  arg_ts = 20 # really want 0-9 + 20 * (arbitrary random 0 to some number)
  expected_num_dots = (arg_ts % 10) + 1
  dots_string = ""

  for i in range(expected_num_dots):
    dots_string = dots_string + '.'

  mocker.patch("builtins.print")

  # Act
  util.sim_io.print_dots(arg_ts)

  # Assert
  print.assert_called_with('\033[95m' + dots_string + '\033[0m')

def test_print_dots_uses_10_minus_mod_10_plus_one_dots_when_ts_mod_20_is_10(mocker):
  # Arrange
  arg_ts = 10 # 10 is a static value by design but should still add 20 * 0 to some number
  expected_num_dots = 10 - (arg_ts % 10) + 1
  dots_string = ""

  for i in range(expected_num_dots):
    dots_string = dots_string + '.'

  mocker.patch("builtins.print")

  # Act
  util.sim_io.print_dots(arg_ts)

  # Assert
  print.assert_called_with('\033[95m' + dots_string + '\033[0m')

def test_print_dots_uses_10_minus_mod_10_plus_one_dots_when_ts_mod_20_is_greater_than_10(mocker):
  # Arrange
  arg_ts = 19 # really should be 11 to 19 + 20 * 0 to some random number
  expected_num_dots = 10 - (arg_ts % 10) + 1
  dots_string = ""

  for i in range(expected_num_dots):
    dots_string = dots_string + '.'

  mocker.patch("builtins.print")

  # Act
  util.sim_io.print_dots(arg_ts)

  # Assert
  print.assert_called_with('\033[95m' + dots_string + '\033[0m')