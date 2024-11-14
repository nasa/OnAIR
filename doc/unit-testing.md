# Unit Testing in OnAIR

## How to Run the Unit Tests

### Prerequisites

Before running the unit tests, ensure you have the following Python packages installed:

Required:
- pytest
- pytest-mock
- coverage

Optional:
- pytest-randomly

You can install these packages using pip:

```
pip install pytest pytest-mock coverage pytest-randomly
```

### Running the Test Suite with driver.py

From the parent directory of your local OnAIR repository:

```
python driver.py -t
```

#### Optional Settings for driver.py

You can add these options to the driver.py test run:

- `--conftest-seed=###`: Set the random values seed for this run
- `--randomly-seed=###`: Set the random order seed for this run, must have randomly installed
- `--verbose` or `-v`: Set verbosity level (incremental, use -vv, -vvv for more verbosity)
- `-k KEYWORD`: Only run tests that match the KEYWORD (see `pytest --help` for more details)

Note: Running tests will output results using provided seeds, but each seed is random when not set directly.

Example start of test output:

```
Using --conftest-seed=1691289424
===== test session starts =======
platform linux -- Python 3.11.2, pytest-7.2.0, pluggy-1.3.0
Using --randomly-seed=1956010105
```

To repeat results, copy and paste the output seeds as arguments in your next run.

### Running pytest Directly from Command Line

For the equivalent of the driver.py run:

```
python -m coverage run --branch --source=onair,plugins -m pytest ./test/
```

#### Command Breakdown:

- `python -m`: Invokes the Python runtime on the library following the -m
- `coverage run`: Runs coverage data collection during testing
- `--branch`: Includes code branching information in the coverage report
- `--source=onair,plugins`: Specifies where the code under test exists for reporting line hits
- `-m pytest`: Tells coverage to use pytest as the test runner
- `./test`: Runs all tests found in this directory and subdirectories

#### Optional Command Line Settings

- `--disable-warnings`: Removes warning reports, but displays count
- `-p no:randomly`: Stops random order testing (only if pytest-randomly is installed)
- `--conftest-seed=###`: Sets the random values seed for this run
- `--randomly-seed=###`: Sets the random order seed for this run
- `--verbose` or `-v`: Sets verbosity level (incremental, use -vv, -vvv for more verbosity)
- `-k KEYWORD`: Only runs tests that match the KEYWORD

### Viewing Test Coverage

After running the tests:

1. For basic results in the terminal:

```
coverage report
```

2. For a detailed HTML report:

```
coverage html
```

   Then open `htmlcov/index.html` in your web browser (e.g., `firefox htmlcov/index.html`)

Note: You may need to use `python -m` before these commands (e.g., `python -m coverage report`).
