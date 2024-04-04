# The On-board Artificial Intelligence Research (OnAIR) Platform

The On-board Artificial Intelligence Research (OnAIR) Platform is a framework that enables AI algorithms written in Python to interact with NASA's [cFS](https://github.com/nasa/cFS).
It is intended to explore research concepts in autonomous operations in a simulated environment.

## Generating environment

Create a conda environment with the necessary packages

    conda create --name onair --file requirements_pip.txt

## Containerized Run

The included [Dockerfile](./Dockerfile) may be used to run OnAIR in a container that is also capable of running cFS.


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

## License and Copyright

Please refer to [NOSA GSC-19165-1 OnAIR.pdf](NOSA%20GSC-19165-1%20OnAIR.pdf) and [COPYRIGHT](COPYRIGHT).

## Contributions

Please open an issue if you find any problems.
We are a small team, but will try to respond in a timely fashion.

If you would like to contribute code, GREAT!
First you will need to complete the [Individual Contributor License Agreement (pdf)](doc/Indv_CLA_OnAIR.pdf) and email it to gsfc-softwarerequest@mail.nasa.gov with james.marshall-1@nasa.gov CCed.

Next, please create an issue for the fix or feature and note that you intend to work on it.
Fork the repository and create a branch with a name that starts with the issue number.
Once done, submit your pull request and we'll take a look.
You may want to make draft pull requests to solicit feedback on larger changes.
