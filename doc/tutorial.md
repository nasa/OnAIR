# Tutorial

This guide will show you how to run OnAIR in its default configuration, explains the configuration options, and add the included Kalman plugin.

Jump to "Tutorial: Using the Kalman Plugin" at the bottom to skip learning about things and just start making changes.

## Installation

You will need a Python environment (we test with 3.9 and newer) and the additional libraries listed in [environment.yml](../environment.yml).
We provide a [Dockerfile](../Dockerfile) which can be used directly to create a container to run OnAIR and also serves to document one possible installation.

Clone the OnAIR repo:

```
> git clone https://github.com/nasa/OnAIR.git OnAIR
> cd OnAIR
```

## First Run: Default Configuration

OnAIR includes some example data files, default configuration, and example plugin.

To run OnAIR:

`OnAIR> python driver.py`

You should see output to the console like this:

```
***************************************************
************    SIMULATION STARTED     ************
***************************************************

--------------------- STEP 1 ---------------------

CURRENT DATA: [946706400.0, 13.0, 3.0, 0.0, 2000.0, 0.0, 34.29, 0.0, 0.0]
INTERPRETED SYSTEM STATUS: ---

--------------------- STEP 2 ---------------------

CURRENT DATA: [946706401.0, 18.0, 5.0, -199.99, 1999.79, -0.41, 34.3, 0.0, 0.0]
INTERPRETED SYSTEM STATUS: ---

--------------------- STEP 3 ---------------------

CURRENT DATA: [946706402.0, 18.0, 5.0, -199.99, 1997.89, -1.8, 34.41, 0.0, 0.0]
INTERPRETED SYSTEM STATUS: ---

...
```

This will run OnAIR with the default configuration which loads a .csv file, steps through each row of data, and does nothing with it.

## Configuration File

The default configuration is [default_config.ini](../onair/config/default_config.ini).
Let's look at some of the parameters it sets.

### Telemetry file

```
TelemetryFilePath = onair/data/raw_telemetry_data/data_physics_generation/Errors
TelemetryFile = 700_crash_to_earth_1.csv
```

The TelemetryFilePath and TelemetryFile parameters tell OnAIR what data to load.
OnAIR is able to parse Comma Separated Value (.csv) files with the assumptions that the first row are the column names and all rows have the same number of cells.
Note that these parameters are ignored and can be left blank if you intend to run OnAIR with a live data source (see [redis_example.ini](../onair/config/redis_example.ini).

Here's a small snippet of [700_crash_to_earth_1.csv](../onair/data/raw_telemetry_data/data_physics_generation/Errors/700_crash_to_earth_1.csv):

```
Time,VOLTAGE,CURRENT,THRUST,ALTITUDE,ACCELERATION,TEMPERATURE,SCIENCE_COLLECTION,[LABEL]: ERROR_STATE
0:00,13,3,0,2000,0,34.29,0,0
0:01,18,5,-199.99,1999.79,-0.41,34.3,0,0
0:02,18,5,-199.99,1997.89,-1.8,34.41,0,0
0:03,18,5,-199.99,1995.73,-2.32,34.53,0,0
```

You'll notice that the ALTITUDE telemetry point eventually becomes 0, thus the file name.
This was an error condition for the simulated rocket 🔥.

### Meta File

```
MetaFilePath = onair/data/telemetry_configs/
MetaFile = data_physics_generation_CONFIG.json
```

Telemetry files provide the raw data for OnAIR to process; the meta files provide information about the raw data.
For example, a voltage telemetry point may have a range of acceptable values, noted as a FEASIBILITY test in this snipped of [data_physics_generation_CONFIG.json](../onair/data/telemetry_configs/data_physics_generation_CONFIG.json):

```
    "POWER": {
      "VOLTAGE": {
        "conversion": "",
        "tests": {
          "FEASIBILITY": "[12.9999, 18.1111]"
        },
        "description": "No description"
      },
```

The meta file also defines the order in which the telemetry points will be received in and additional information for live data sources such as the messages or channels that OnAIR needs to subscribe to for data.

### Data Source

`DataSourceFile = onair/data_handling/csv_parser.py`

This line defines the [on_air_data_source.py](../onair/data_handling/on_air_data_source.py) that will be used to ingest data into OnAIR.
The [csv_parser.py](../onair/data_handling/csv_parser.py) is a file-based data source that is used by the default configuration; [redis_adapter.py](../onair/data_handling/redis_adapter.py) and [sbn_adapter.py](../onair/data_handling/sbn_adapter.py) are live data sources that connect to an external publish/subscribe service to ingest data.
File-based sources are intended to process saved telemetry and allow for testable, repeatable experiments that are independent of control software.
Live sources are intended to allow OnAIR to pull data from a running system such as [NASA's cFS](https://github.com/nasa/cFS).

### Plugins

Finally we reach the plugin section which is at the heart of OnAIR.
This is where researchers can insert their own code to process data and generate higher level data products.
There are four plugin types, of which there can be multiple plugins instantiated.
Data is passed from OnAIR to the plugins as a frame of telemetry points at each time step, and data is passed between plugins as Python dictionaries.

Refer to the [Architecture](architecture.md) guide for more information.

In the default configuration, only a generic plugin that does no processing is used:

```
KnowledgeRepPluginDict = {'generic':'plugins/generic/__init__.py'}
LearnersPluginDict = {'generic':'plugins/generic/__init__.py'}
PlannersPluginDict = {'generic':'plugins/generic/__init__.py'}
ComplexPluginDict = {'generic':'plugins/generic/__init__.py'}
```

## Tutorial: Using the Kalman Plugin
