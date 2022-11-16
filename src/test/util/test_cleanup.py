import pytest
from mock import MagicMock
import util.cleanup

# clean_all tests
def test_clean_all_removes_provided_file_types_back_two_directories_from_file_location_when_given_run_path_is_empty_string(mocker):
  # Arrange
  arg_run_path = ''

  fake_path = MagicMock()
  fake_run_path = str(MagicMock())

  mocker.patch('util.cleanup.os.path.realpath', return_value=fake_path)
  mocker.patch('util.cleanup.os.path.dirname', return_value=fake_run_path)
  mocker.patch('util.cleanup.os.chdir')
  mocker.patch('util.cleanup.os.system')

  # Act
  util.cleanup.clean_all(arg_run_path)

  # Assert
  assert util.cleanup.os.path.realpath.call_count == 1
  assert util.cleanup.os.path.dirname.call_count == 1
  assert util.cleanup.os.chdir.call_count == 1
  assert util.cleanup.os.chdir.call_args_list[0].args == (fake_run_path + '/../../',)
  assert util.cleanup.os.system.call_count == 2
  assert util.cleanup.os.system.call_args_list[0].args == ('find . | grep -E "(__pycache__|.pyc|.pyo$)" | xargs rm -rf', )
  assert util.cleanup.os.system.call_args_list[1].args == ('find . | grep -E ".DS_Store" | xargs rm -rf', )

def test_clean_all_removes_provided_file_types_from_given_run_path_when_it_is_not_empty(mocker):
  # Arrange
  arg_run_path = str(MagicMock())

  fake_path = MagicMock()
  fake_run_path = str(MagicMock())

  mocker.patch('util.cleanup.os.path.realpath', return_value=fake_path)
  mocker.patch('util.cleanup.os.path.dirname', return_value=fake_run_path)
  mocker.patch('util.cleanup.os.chdir')
  mocker.patch('util.cleanup.os.system')

  # Act
  util.cleanup.clean_all(arg_run_path)

  # Assert
  assert util.cleanup.os.path.realpath.call_count == 0
  assert util.cleanup.os.path.dirname.call_count == 0
  assert util.cleanup.os.chdir.call_count == 1
  assert util.cleanup.os.chdir.call_args_list[0].args == (arg_run_path, )
  assert util.cleanup.os.system.call_count == 2
  assert util.cleanup.os.system.call_args_list[0].args == ('find . | grep -E "(__pycache__|.pyc|.pyo$)" | xargs rm -rf', )
  assert util.cleanup.os.system.call_args_list[1].args == ('find . | grep -E ".DS_Store" | xargs rm -rf', )


# test_setup_folders
def test_setup_folders_creates_dir_when_given_results_path_does_not_exist(mocker):
  # Arrange
  arg_results_path = str(MagicMock())

  mocker.patch('util.cleanup.os.path.isdir', return_value=False)
  mocker.patch('util.cleanup.os.mkdir')

  # Act
  util.cleanup.setup_folders(arg_results_path)
  
  # Assert
  assert util.cleanup.os.path.isdir.call_count == 1
  assert util.cleanup.os.mkdir.call_count == 1
  assert util.cleanup.os.mkdir.call_args_list[0].args == (arg_results_path, )

def test_setup_folders_does_not_create_dir_when_it_already_exists(mocker):
  # Arrange
  arg_results_path = str(MagicMock())

  mocker.patch('util.cleanup.os.path.isdir', return_value=True)
  mocker.patch('util.cleanup.os.mkdir')

  # Act
  util.cleanup.setup_folders(arg_results_path)
  
  # Assert
  assert util.cleanup.os.path.isdir.call_count == 1
  assert util.cleanup.os.mkdir.call_count == 0