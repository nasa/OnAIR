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

## Running test_all

Make sure to set the environment variables either before on run or previously 

    RESULTS_PATH=results RUN_PATH=results python test_all.py
