![Build](https://github.com/nasa/OnAIR/actions/workflows/unit-test.yml/badge.svg)
[![CodeCov](https://codecov.io/gh/nasa/OnAIR/branch/main/graph/badge.svg?token=L0WVOTD5X9)](https://codecov.io/gh/nasa/OnAIR)

![alt text](OnAIR_logo.svg "The OnAIR logo, italicized NASA worm style font in blue and orange")

# The On-board Artificial Intelligence Research (OnAIR) Platform

The On-board Artificial Intelligence Research (OnAIR) Platform is a framework that enables AI algorithms written in Python to interact with NASA's [cFS](https://github.com/nasa/cFS).
It is intended to explore research concepts in autonomous operations in a simulated environment.

## Generating environment

Create a conda environment with the necessary packages

    conda create --name onair --file requirements_pip.txt

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

#### A few optional settings for the driver.py file
Options that may be added to the driver.py test run. Use these at your own discretion.

`--conftest-seed=###` - set the random values seed for this run  
`--randomly-seed=###` - set the random order seed for this run  
`--verbose` or `-v` - set verbosity level, also -vv, -vvv, etc.  

NOTE: Running tests will output results using provided seeds, but each seed is random when not set directly.  
Example start of test output:
```
Using --conftest-seed=1691289424
===== test session starts =======
platform linux -- Python 3.11.2, pytest-7.2.0, pluggy-1.3.0
Using --randomly-seed=1956010105
```
Copy and paste previously output seeds (or type them out) as the arguments to repeat results.

### Running pytest directly from command line

For the equivalent of the driver.py run:
```
python -m coverage run --branch --source=onair,plugins -m pytest ./test/
```

#### Command breakdown:

`python -m` - invokes the python runtime on the library following the -m  
`coverage run` - runs coverage data collection during testing, wrapping itself on the test runner used  
`--branch` - includes code branching information in the coverage report  
`--source=onair,plugins` - tells coverage where the code under test exists for reporting line hits  
`-m pytest` - tells coverage what test runner (framework) to wrap  
`./test` - run all tests found in this directory and subdirectories  

#### A few optional settings for the command line
Options that may be added to the command line test run. Use these at your own discretion.

`--disable-warnings` - removes the warning reports, but displays count (i.e., 124 passed, 1 warning in 0.65s)  
`-p no:randomly` - ONLY required to stop random order testing IFF pytest-randomly installed
`--conftest-seed=###` - set the random values seed for this run  
`--randomly-seed=###` - set the random order seed for this run  
`--verbose` or `-v` - set verbosity level, also -vv, -vvv, etc.  

NOTE: see note about seeds in driver.py section above

### To view testing line coverage after run:
NOTE: you may or may not need the `python -m` to run coverage report or html

`coverage report` - prints basic results in terminal  
or
`coverage html` - creates htmlcov/index.html, automatic when using driver.py for testing

then  
`<browser_here> htmlcov/index.html` - browsable coverage (i.e., `firefox htmlcov/index.html`)

## License and Copyright

Please refer to [NOSA GSC-19165-1 OnAIR.pdf](NOSA%20GSC-19165-1%20OnAIR.pdf) and [COPYRIGHT](COPYRIGHT).

## Contributions

Please open an issue if you find any problems.
We are a small team, but will try to respond in a timely fashion.

If you would like to contribute to the repository, GREAT!
First you will need to complete the [Individual Contributor License Agreement (pdf)](doc/Indv_CLA_OnAIR.pdf).
Then, email it to gsfc-softwarerequest@mail.nasa.gov with james.marshall-1@nasa.gov CCed.

Next, please create an issue for the fix or feature and note that you intend to work on it.
Fork the repository and create a branch with a name that starts with the issue number.
Once done, submit your pull request and we'll take a look.
You may want to make draft pull requests to solicit feedback on larger changes.
