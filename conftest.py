# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

import pytest
import random
import datetime
from mock import MagicMock
import sys

def pytest_configure():
  seed = datetime.datetime.now().timestamp()
  pytest.gen = random.Random(seed)
  print(f"conftest random seed = {seed}")

  # Mock simdkalman for kalman_plugin testing
  simdkalman = MagicMock()
  sys.modules['simdkalman'] = simdkalman

  # Mock sbn_client for sbn_adapter testing
  sc = MagicMock()
  sys.modules['sbn_client'] = sc

  # Mock message_headers for sbn_adapter testing
  mh = MagicMock()
  mh.sample_data_tlm_t = MagicMock()
  mh.sample_data_tlm_t.__name__ = 'mock_sample_data_tlm_t'
  mh.sample_data_power_t = MagicMock()
  mh.sample_data_power_t.__name__ = 'mock_sample_data_power_t'
  mh.sample_data_thermal_t = MagicMock()
  mh.sample_data_thermal_t.__name__ = 'mock_sample_data_thermal_t'
  mh.sample_data_gps_t = MagicMock()
  mh.sample_data_gps_t.__name__ = 'mock_sample_data_gps_t'
  sys.modules['message_headers'] = mh
