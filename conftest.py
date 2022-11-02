import pytest
import random
import datetime

def pytest_configure():
  seed = datetime.datetime.now().timestamp()
  pytest.gen = random.Random(seed)
  print(f"conftest random seed = {seed}")