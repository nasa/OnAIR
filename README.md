# The On-board Artificial Intelligence Research (OnAIR) Platform

## Generating environment

Create a conda environment with the necessary packages

    conda create --name onair --file requirements.txt

## Running unit tests

Instructions on how to run unit tests for OnAIR

### Required python installs:
pytest,
pytest-mock,
coverage

### Optional python install:
pytest-randomly

### Running the unit tests from the driver.py file

From the parent directory of your local repository:
```
python driver.py -t
```

### Running pytest directly from command line

For the equivalent of the driver.py run:
```
python -m coverage run --branch --source=src,data_handling,utils -m pytest ./test/
```

#### Command breakdown:
`python -m` - invokes the python runtime on the library following the -m

`coverage run` - runs coverage data collection during testing, wrapping itself on the test runner used

`--branch` - includes code branching information in the coverage report

`--source=src,data_handling,utils` - tells coverage where the code under test exists for reporting line hits

`-m pytest` - tells coverage what test runner (framework) to wrap

`./test` - run all tests found in this directory and subdirectories

#### A few optional settings
Options that may be added to the command line test run. Use these at your own discretion.

`PYTHONPATH=src` - sets env variable so tests can find src, but only use if tests won't run without

`--disable-warnings` - removes the warning reports, but displays count (i.e., 124 passed, 1 warning in 0.65s)

`-p no:randomly` - ONLY required to stop random order testing IFF pytest-randomly installed

### To view testing line coverage after run:

`coverage report` - prints basic results in terminal

or

`coverage html` - creates htmlcov/index.html, automatic when using driver.py for testing

and

`<browser_here> htmlcov/index.html` - browsable coverage (i.e., `firefox htmlcov/index.html`)