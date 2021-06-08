# RAISR-2.0

Hi James! Surprise... !!!!

Re-writing a fully tested and rearchitected system. 

TODO: Multi file source not currently working for 42

UNIT TESTS: To be Completed
Parser Util
Execution Engine 
Sim 
Driver
print_io 
file_io
feasibility test, double check others ... 
sim_io

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