import pytest
import random
import datetime
from mock import MagicMock
import sys

def pytest_configure():
  seed = datetime.datetime.now().timestamp()
  pytest.gen = random.Random(seed)
  print(f"conftest random seed = {seed}")

  simdkalman = MagicMock()
  sys.modules['simdkalman']  =simdkalman