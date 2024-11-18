![Python Versions](https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10%20|%203.11%20|%203.12-blue.svg)
![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)
![Build](https://github.com/nasa/OnAIR/actions/workflows/unit-test.yml/badge.svg)
[![CodeCov](https://codecov.io/gh/nasa/OnAIR/branch/main/graph/badge.svg?token=L0WVOTD5X9)](https://codecov.io/gh/nasa/OnAIR)

![alt text](doc/images/OnAIR_logo.svg "The OnAIR logo, italicized NASA worm style font in blue and orange")

# The On-board Artificial Intelligence Research (OnAIR) Platform

The On-board Artificial Intelligence Research (OnAIR) Platform is a framework that enables AI algorithms written in Python to interact with NASA's [cFS](https://github.com/nasa/cFS).
It is intended to explore research concepts in autonomous operations in a simulated environment.
Featuring a plugin style architechture, it is designed to facilitate rapid prototyping of cognitive agents.

## Key Features

- Python-based AI algorithm support
- Configurable to use different data source types
  - Core Flight System (cFS)
  - CSV file
  - Redis
- Adaptable to new data source types
  - Selectable at runtime
  - Set your own data source
- Cognitive architechture based data pipeline
  - Knowledge Representations
  - Learners
  - Planners
  - Complex Reasoners
- Flexible plugin system for cognitive components

## Quick Start

**NOTE:** You may need to put your specific python or version in place of `python`, e.g., `python3.12`

**NOTE:** You may need the `-m` option to run python modules, e.g., `python -m coverage report`

### 1. Clone the repository:
```
git clone https://github.com/nasa/OnAIR.git
cd OnAIR
```
Gets the repository and enters the directory.

### 2. Set up the environment:
Using your local Python installation (your `pip` command may vary, e.g., `pip3.12`)
```
pip install -r requirements.txt
```
This installs the packages necessary for testing and running.

### 3. Run the unit tests and check the coverage:
NOTE: this step is technically optional, but highly recommended.
```
python driver.py -t
coverage report
```
If all tests pass and coverage is shown to be at 100%, your setup is likely able to use OnAIR.

### 4. Run the default configuration for OnAIR:
```
python driver.py
```
Output should begin. It will probably be very fast, but should look similar to:
```
***************************************************
************    SIMULATION STARTED     ************
***************************************************

--------------------- STEP 1 ---------------------

CURRENT DATA: [946706400.0, 13.0, 3.0, 0.0, 2000.0, 0.0, 34.29, 0.0, 0.0]
INTERPRETED SYSTEM STATUS: ---

--------------------- STEP 2 ---------------------
```
...
```
--------------------- STEP 1439 ---------------------

CURRENT DATA: [946707838.0, 30.0, 12.0, 200.0, 0.0, 0.0, 182.1, 0.0, 1.0]
INTERPRETED SYSTEM STATUS: ---

--------------------- STEP 1440 ---------------------

CURRENT DATA: [946707839.0, 30.0, 12.0, 200.0, 0.0, 0.0, 182.28, 0.0, 1.0]
INTERPRETED SYSTEM STATUS: ---
```
At this point it should stop.
This default used a CSV file as the data source and stops at end of file.
The configuration file used is [default_config.ini](onair/config/default_config.ini).
When this runs without errors, your basic setup is complete.

### 5. Next steps:
#### [Documents and Guides](doc/README.md)
Information on how OnAIR is set up and operates.
#### [Initialization Files](onair/config)
Examples of how to configure OnAIR for use.
#### [Telemetry Definition Files](onair/data/telemetry_configs)
Example setup files for describing the pipeline's data frame.
#### [Plugins](plugins)
The provided cognitive components.
#### [Data Sources](onair/data_handling)
Provided data handling for telemetry input.

## Contributing

Please open an issue if you find any problems.
We are a small team, but will try to respond in a timely fashion.

If you would like to contribute to the repository, GREAT!
First you will need to complete the [Individual Contributor License Agreement (pdf)](doc/Indv_CLA_OnAIR.pdf).
Then, email it to gsfc-softwarerequest@mail.nasa.gov with james.marshall-1@nasa.gov CCed.
Please include your github username in the email.

Next, please create an issue for the fix or feature and note that you intend to work on it.
Fork the repository and create a branch with a name that starts with the issue number.
Once done, submit your pull request and we'll take a look.
You may want to make draft pull requests to solicit feedback on larger changes.

## License and Copyright

Please refer to [NOSA GSC-19165-1 OnAIR.pdf](NOSA%20GSC-19165-1%20OnAIR.pdf) and [COPYRIGHT](COPYRIGHT).
