# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"
import pytest
from mock import MagicMock

import onair.src.util.cleanup as cleanup

# test_setup_folders
def test_cleanup_setup_folders_creates_dir_when_given_results_path_does_not_exist(mocker):
  # Arrange
  arg_results_path = str(MagicMock())

  mocker.patch(cleanup.__name__ + '.os.path.isdir', return_value=False)
  mocker.patch(cleanup.__name__ + '.os.mkdir')

  # Act
  cleanup.setup_folders(arg_results_path)
  
  # Assert
  assert cleanup.os.path.isdir.call_count == 1
  assert cleanup.os.mkdir.call_count == 1
  assert cleanup.os.mkdir.call_args_list[0].args == (arg_results_path, )

def test_cleanup_setup_folders_does_not_create_dir_when_it_already_exists(mocker):
  # Arrange
  arg_results_path = str(MagicMock())

  mocker.patch(cleanup.__name__ + '.os.path.isdir', return_value=True)
  mocker.patch(cleanup.__name__ + '.os.mkdir')

  # Act
  cleanup.setup_folders(arg_results_path)
  
  # Assert
  assert cleanup.os.path.isdir.call_count == 1
  assert cleanup.os.mkdir.call_count == 0
