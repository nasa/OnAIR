import pytest
from mock import MagicMock
import src.util.cleanup

# test_setup_folders
def test_cleanup_setup_folders_creates_dir_when_given_results_path_does_not_exist(mocker):
  # Arrange
  arg_results_path = str(MagicMock())

  mocker.patch('src.util.cleanup.os.path.isdir', return_value=False)
  mocker.patch('src.util.cleanup.os.mkdir')

  # Act
  src.util.cleanup.setup_folders(arg_results_path)
  
  # Assert
  assert src.util.cleanup.os.path.isdir.call_count == 1
  assert src.util.cleanup.os.mkdir.call_count == 1
  assert src.util.cleanup.os.mkdir.call_args_list[0].args == (arg_results_path, )

def test_cleanup_setup_folders_does_not_create_dir_when_it_already_exists(mocker):
  # Arrange
  arg_results_path = str(MagicMock())

  mocker.patch('src.util.cleanup.os.path.isdir', return_value=True)
  mocker.patch('src.util.cleanup.os.mkdir')

  # Act
  src.util.cleanup.setup_folders(arg_results_path)
  
  # Assert
  assert src.util.cleanup.os.path.isdir.call_count == 1
  assert src.util.cleanup.os.mkdir.call_count == 0
