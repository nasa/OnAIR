from mock import patch
import util.sim_io

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