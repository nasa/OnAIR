# RAISR-2.0

## Generating environment

Create a conda environment with the necessary packages

    conda create --name raisr --file requirements.txt

## Running driver file test

Make a results directory in your root folder and test folder:

    mkdir results
    mkdir src/test/results

Move test data from to src folder, and the config to src, unless you already have data there

    cp -r src/test/data src/
    mv src/data/config src

Then you can just run the driver

    python driver.py -t

## Running pytest
How to run unit tests for RAISR

### Required python installs:
pytest,
pytest-mock,
coverage

### Optional python install:
pytest-randomly

### Command used to run the tests:
`PYTHONPATH=src RUN_PATH=./src/test RESULTS_PATH=./src/test/results coverage run --omit="src/test/*" -m pytest ./src/test/`

# Command breakdown:

`PYTHONPATH=src` - sets env variable so tests can find src

`RUN_PATH=./src/test` - sets env variable otherwise several tests will fail because production code needs this set

`RESULTS_PATH=./src/test/results` - Used in production code, MUST create this directory prior to use (see note above) but good to go thereafter

`coverage run` - runs coverage data collection during test wrapping pytest

`--omit="src/test/*"` - keeps the test files out of the coverage statistics

`-m pytest` - tells coverage what test runner (framework) to wrap

`./src/test` - run all tests found in this directory and subdirectories

### A few optional settings
These options may be added to the test run to stop some features from operating. Use these at your own discretion.

`--disable-warnings` - removes the full warning reports, but still counts them (i.e., 124 passed, 1 warning in 0.65s)

`-p no:randomly` - ONLY required to stop random order testing IFF pytest-randomly installed

### To get coverage after run:

`coverage report` - prints basic results in terminal

or

`coverage html` - creates htmlcov/index.html for use by your favorite browser
