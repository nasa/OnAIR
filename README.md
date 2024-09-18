![Build](https://github.com/nasa/OnAIR/actions/workflows/unit-test.yml/badge.svg)
[![CodeCov](https://codecov.io/gh/nasa/OnAIR/branch/main/graph/badge.svg?token=L0WVOTD5X9)](https://codecov.io/gh/nasa/OnAIR)

![alt text](OnAIR_logo.svg "The OnAIR logo, italicized NASA worm style font in blue and orange")

# The On-board Artificial Intelligence Research (OnAIR) Platform

The On-board Artificial Intelligence Research (OnAIR) Platform is a framework that enables AI algorithms written in Python to interact with NASA's [cFS](https://github.com/nasa/cFS).
It is intended to explore research concepts in autonomous operations in a simulated environment. Featuring a plugin style architechture, it is designed to facilitate rapid prototyping of cognitive agents.

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

**NOTE:** You may need to put your specific python version in place of `python`, e.g., `python3.12`  
**NOTE:** You may need the `-m` option to run python modules, e.g., `python -m coverage report`

### 1. Clone the repository:  
```
git clone https://github.com/nasa/OnAIR.git 
cd OnAIR
```
Gets the repository and enters the directory.  

### 2. Set up the environment:
- **Option A**: Using your local Python installation
```
pip install -r requirements.txt
```
- **Option B**: Using Conda
```
conda env create -f environment.yml
conda activate onair-env
```
- **Option C**: Whatever works for you!  

### 3. Run the unit tests:
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
At this point it should stop. This default used a CSV file as the data source and stops at end of file. The configuration file used is [default_config.ini](https://github.com/nasa/OnAIR/blob/main/onair/config/default_config.ini). When this runs without errors, your basic setup is complete.

### 5. Next steps:
Explore the [config directory](https://github.com/nasa/OnAIR/tree/main/onair/config) for example configuration files.
Start developing by plugging in your data source and your own cognitive components.
Refer to the Documentation section for more detailed information on customization and advanced features.

## User Guide

** Under Construction, please pardon our dust. **

### Welcome, You Are Now Live with $\color{blue}{On}\color{orange}{AI}\color{blue}{R}$!
  - What is OnAIR?
  - More Importantly, What Isn't OnAIR?
  - How to Use This Guide
### Detailed Installation
  - Requirements and Dependencies
  - Using pip
  - Using Conda
  - Troubleshooting
### Unit Testing
  - Philosopy
  - How to Run the Unit Tests
  - How to Read a Unit Test
  - How to Write a Unit Test for OnAIR
### Configuration
  - Configuration File Structure (INI)
  - Meta Data File Structure (JSON)
### Data Sources
  - Provided Examples
    - Core Flight System (cFS)
    - CSV Files
    - Redis
  - Developing Your Own Data Sources
  - How to Attach a Data Source
### The Cognitive Components
  - Knowledge Representation
  - Learner
  - Planner
  - Complex Reasoner
  - Data Flow Through Components 
### Plugins
  - Plugin Development
  - How to Attach Your Plugin
### Provided Usage Examples
  - The Default
  - The Reporter
  - The cFS?
  - The Redis
### Advanced Usage Examples
  - A Mock Mission
### Troubleshooting
  - Error Outputs During Use
  - Debugging
  - Issue Reporting

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

## Contact

[Placeholder for contact information]
