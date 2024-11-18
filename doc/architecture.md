
# **WIP: Under Construction and not yet complete.**

The cognitive architecture is the backbone of OnAIR.
The configuration files provide the structure that OnAIR is to use at runtime.
They set up the data source, the conduit for telemetry to enter the system, and the pipeline, the series of cognitive components that run.

- [Configuration Files](#the-configuration-files)
  - [Initialization File](#initialization-ini-file)
  - [Telemetry Definition File](#telemetry-definition-json-file)
- [OnAirDataSource](#the-onairdatasource)
  - [CSV Parser](#csv-parser)
  - [Redis Adapter](#redis-adapter)
  - [SBN Adapter](#sbn-adapter)
- [Pipeline](#the-pipeline)
  - [Data Types](#data-types)
  - [Plugins](#the-plugins)
  - [Flow](#flow)

## The Configuration Files
These files define the components and data structure that will be used by a running instance of OnAIR.

### Initialization (.ini) File
Provides the details about the components and other files used by OnAIR when performing its data reasoning.

>#### **[FILES]** (Required)
>Pathing and file name information
>|Key|Required?|Expected Type|Description|
>|-|-|-|-|
>|TelemetryFilePath|Yes|str|location of the telemetry data file on the local system|
>|TelemetryFile|Yes|str|name of the telemetry data file, used by [OnAir data sources](#the-onairdatasource) that access files|
>|MetaFilePath|Yes|str|location of the [telemetry definitions file](#telemetry-definitions-json-file)|
>|MetaFile|Yes|str|name of the [telemetry definitions file](#telemetry-definitions-json-file), a JSON file|

>#### **[DATA_HANDLING]** (Required)
>The telemetry data parser
>|Key|Required?|Expected Type|Description|
>|-|-|-|-|
>|DataSourceFile|Yes|str|full path and python file name of the [OnAir data source](#the-onairdatasource) to use|

>#### **[PLUGINS]** (Required)
>The various [plugins](#the-plugins) that OnAIR will use to construct the [pipeline](#the-pipeline)
>|Key|Required?|Expected Type|Description|
>|-|-|-|-|
>|KnowledgeRepPluginDict|Yes|dict|Knowledge Representation plugins to be used at runtime|
>|LearnersPluginDict|Yes|dict|Learner plugins to be used at runtime|
>|PlannersPluginDict|Yes|dict|Planner plugins to be used at runtime|
>|ComplexPluginDict|Yes|dict|Complex Reasoner plugins to be used at runtime|
>
>Each plugin dictionary item consists of the key, name for the plugin at runtime, to value, path to the plugin's base directory.
Runtime plugin name is not required to be the same as the name of the plugin, but the directory used for the plugin must contain the concrete plugin class named **\<directory name\>_plugin.py**.
Multiple of the same plugin directory may be used by defining unique key names, e.g. {"Reporter1":"plugins/reporter", "Reporter2":"plugins/reporter"} will provide two reporter plugins.
A plugin type may be left empty by giving it an empty dict, **{}**.

>#### **[OPTIONS]** (Optional)
>Provides settings for runtime
>|Key|Required?|Expected Type|Description|
>|-|-|-|-|
>|IO_Enabled| No | bool | runtime text output or not, omission equates to **_false_**

### Telemetry Definition (.json) File
WIP - Section incomplete

Defined in the [initialization file](#initialization-ini-file) as the 'Metafile.'
Provides the information necessary for OnAIR to understand received data.

## The OnAirDataSource
This is the adapter that attaches OnAIR to a telemetry source.
It provides the basic data that is used by the cognitive components to render reasoning about the system.

There are a few provided by default (located in the [onair/data_handling](../onair/data_handling) directory):
- CSV - data from a .csv formatted file
- Redis - data from Redis server channels
- SBN - data via the Software Bus Network attached to an instance of [NASA's core Flight Software](https://github.com/nasa/cFS)

However, all of these just implement the abstract class [`OnAirDataSource`](../onair/data_handling/on_air_data_source.py) and users are able (nay, encouraged!) to write their own.

### [CSV Parser](../onair/data_handling/csv_parser.py)
When you have telemetry data in CSV (Comma-Separated Values) format that you want to use with the OnAIR platform.

#### CSV File Requirements
What a file must have to be interpreted correctly.

- The first row shall contain headers that match the order from the [telemetry definition file](#telemetry-definition-json-file)
- Each subsequent row shall contain the values that represent a single frame of data

#### CSV File Considerations
What should be known about file parsing and usage.

- Acceptable values, all will become floats
  - Integer
  - Float
  - Timestamp str
- All other values are unacceptable and turned into 0.0 (by the [floatify_input](../onair/data_handling/parser_util.py#L49) method)
- Parser loads the entire floatified CSV file into memory at once.
- Memory usage increases with file size

#### CSV Potential Updates
Ideas for future expandability, alternative designs, or user development.

- Allow for multiple files
  - chained in sequence
  - dynamically selected
- Line by line file access to lower memory usage

### [Redis Adapter](../onair/data_handling/redis_adapter.py)
WIP - Section incomplete

### [SBN Adapter](../onair/data_handling/sbn_adapter.py)
WIP - Section incomplete

## The Pipeline
This is the path in which the received data from the source will flow.

The order of plugin operations is determined by placement in the plugin list within the [initilization file](#initialization-ini-file) used when running OnAIR.
Plugins run from first to last in the order they are placed within their dictionary definitions.

Data received by the defined [OnAirDataSource](#the-onairdatasource) is fed into the pipeline in a by taking the currently received telemetry, referred to as the frame, which gets transferred to the pipeline into a Python list called the `low_level_data`.
Each frame is like a cell in animation, with only the new data received layered on top of old data.
Stagnant data is kept in the `low_level_data`, much like the background in an animated feature doesn't change; whereas current frame data overwrites any old data, like characters in the foreground moving about.
The new data replaces old data, but unreceived telemetry values are retained.

For instance:
>Our data order of telemetry is [Time, Money, Effort]. Time will increment at every step, '-' means no data.
>
>| Step | data | frame | low_level_data | Notes |
>|-|-|-|-|-|
>|0| Init | ['-', '-', '-'] | ['-', '-', '-'] | No data filled |
>|1| None | [0, '-', '-'] | [0, '-', '-'] | Only Time is set |
>|2| Effort, 5 | [1, '-', 5] | [1, '-', 5] | Time overlain, Effort received |
>|3| Effort, 10 | [2, '-', 10] | [2, '-', 10] | Time and Effort overlain |
>|4| Money, 20 | [3, 20, '-'] | [3, 20, 10] | Time overlain, Money received, Effort retained |
>|5| Money, 40 | [4, 40, 0] | [4, 40, 0] | Time, Money and Effort overlain |
>| | Effort, 0 | | | |
>|6| None | [5, '-', '-'] | [5, 40, 0] | Time overlain, Money and Effort retained |


### Data Types
There are two types of data used by the constructs in OnAIR, low and high level data

- `low_level_data`
  - Python list that contains telemetry values as floats
  - Updated with new telemetry values from the frame at each step of operation

- `high_level_data`
  - A nested dictionary structure that stores the output of each plugin's `render_reasoning` method
  - Organized by plugin types:
    - 'vehicle_rep': Knowledge Representation plugins
    - 'learning_systems': Learner plugins
    - 'planning_systems': Planner plugins
    - 'complex_systems': Complex Reasoner plugins
  - For each plugin type, there's a sub-dictionary where:
    - Keys are the plugin runtime names
    - Values are the outputs from their respective `render_reasoning` methods
  - This structure is built and expanded as the pipeline progresses

### The Plugins

The plugins in OnAIR operate on the data types described above.
There are two methods required to be implemented by OnAIR for the plugins:

- `update`: Pulls data into the plugin for use.
This includes either `low_level_data`, `high_level_data`, or both, depending on the plugin type.
- `render_reasoning`: Pushes out reasoned results data.
This is the data you want made available to plugins further down the pipeline.

What a plugin does with the data and how it renders its reasoning is up to the plugin developer to decide.
Reasoned results can be of any type.
However, the order of operations is established by the initialize file used and OnAIR's plugin flow.

### Flow

The pipeline processes data through different types of plugins in a specific order. After each plugin type runs its `render_reasoning` method, the results are stored in the `high_level_data` dictionary. For each plugin, its name becomes the key, and its returned reasoning value becomes the value in the appropriate sub-dictionary of `high_level_data`.

The flow of data through the pipeline is as follows:

- Knowledge Representations (KR) - run first
  - all KRs run `update`, receiving ONLY `low_level_data`
  - all KRs run `render_reasoning`
  - KR reasonings are added to `high_level_data['vehicle_rep']`

- Learners (LN) - run second
  - all LNs run `update`, receiving `low_level_data` and all KR's reasonings in the `high_level_data`
  - LNs do not receive other LNs reasonings
  - all LNs run `render_reasoning`
  - LN reasonings are added to `high_level_data['learning_systems']`

- Planners (PL) - run third
  - all PLs run `update`, receiving ONLY KR's and LN's reasonings in the `high_level_data`
  - PLs do not receive other PLs reasonings
  - all PLs run `render_reasoning`
  - PL reasonings are added to `high_level_data['planning_systems']`

- Complex Reasoners (CR) - run last
  - each CR runs `update` and `render_reasoning` receiving all KR, LN, PL, and any previous CR's reasonings in the `high_level_data`
  - CR reasonings are added to `high_level_data['complex_systems']`
  - Each CR in turn gets progressively more data than the one before it



Here is a illustration of this relationship:

![Pipeline Data Flow](images/OnAIR_pipeline_data_flow.png)


The best way to see this relationship in action is to run the [reporter example](../onair/config/reporter_config.ini):
```
python driver.py onair/config/reporter_config.ini
```
Sample output:
```
...
INTERPRETED SYSTEM STATUS: ---
Knowledge Reporter 1: UPDATE
Knowledge Reporter 2: UPDATE
Knowledge Reporter 1: RENDER_REASONING
Knowledge Reporter 2: RENDER_REASONING
Learners Reporter 1: UPDATE
Learners Reporter 2: UPDATE
Learners Reporter 1: RENDER_REASONING
Learners Reporter 2: RENDER_REASONING
Planner Reporter 1: UPDATE
Planner Reporter 2: UPDATE
Planner Reporter 1: RENDER_REASONING
Planner Reporter 2: RENDER_REASONING
Complex Reporter 1: UPDATE
Complex Reporter 1: RENDER_REASONING
Complex Reporter 2: UPDATE
Complex Reporter 2: RENDER_REASONING
...
```

This gives some minimal output; however, you can edit the [reporter plugin](../plugins/reporter/reporter_plugin.py#L14) line 14 and change verbose_mode to True.
This makes the reporter plugin useful as a debug tool.

After the change, run and you should see something like this:

```
...
CURRENT DATA: [946707838.0, 30.0, 12.0, 200.0, 0.0, 0.0, 182.1, 0.0, 1.0]
INTERPRETED SYSTEM STATUS: ---
Knowledge Reporter 1: UPDATE
 : headers ['Time', 'VOLTAGE', 'CURRENT', 'THRUST', 'ALTITUDE', 'ACCELERATION', 'TEMPERATURE', 'SCIENCE_COLLECTION', 'LABEL_ERROR_STATE']
 : low_level_data <class 'list'> = '[946707839.0, 30.0, 12.0, 200.0, 0.0, 0.0, 182.28, 0.0, 1.0]'
 : high_level_data <class 'dict'> = '{}'
Knowledge Reporter 2: UPDATE
 : headers ['Time', 'VOLTAGE', 'CURRENT', 'THRUST', 'ALTITUDE', 'ACCELERATION', 'TEMPERATURE', 'SCIENCE_COLLECTION', 'LABEL_ERROR_STATE']
 : low_level_data <class 'list'> = '[946707839.0, 30.0, 12.0, 200.0, 0.0, 0.0, 182.28, 0.0, 1.0]'
 : high_level_data <class 'dict'> = '{}'
Knowledge Reporter 1: RENDER_REASONING
 : My low_level_data is [946707839.0, 30.0, 12.0, 200.0, 0.0, 0.0, 182.28, 0.0, 1.0]
 : My high_level_data is {}
Knowledge Reporter 2: RENDER_REASONING
 : My low_level_data is [946707839.0, 30.0, 12.0, 200.0, 0.0, 0.0, 182.28, 0.0, 1.0]
 : My high_level_data is {}
Learners Reporter 1: UPDATE
 : headers ['Time', 'VOLTAGE', 'CURRENT', 'THRUST', 'ALTITUDE', 'ACCELERATION', 'TEMPERATURE', 'SCIENCE_COLLECTION', 'LABEL_ERROR_STATE']
 : low_level_data <class 'list'> = '[946707839.0, 30.0, 12.0, 200.0, 0.0, 0.0, 182.28, 0.0, 1.0]'
 : high_level_data <class 'dict'> = '{'vehicle_rep': {'Knowledge Reporter 1': None, 'Knowledge Reporter 2': None}}'
Learners Reporter 2: UPDATE
 : headers ['Time', 'VOLTAGE', 'CURRENT', 'THRUST', 'ALTITUDE', 'ACCELERATION', 'TEMPERATURE', 'SCIENCE_COLLECTION', 'LABEL_ERROR_STATE']
 : low_level_data <class 'list'> = '[946707839.0, 30.0, 12.0, 200.0, 0.0, 0.0, 182.28, 0.0, 1.0]'
 : high_level_data <class 'dict'> = '{'vehicle_rep': {'Knowledge Reporter 1': None, 'Knowledge Reporter 2': None}}'
Learners Reporter 1: RENDER_REASONING
 : My low_level_data is [946707839.0, 30.0, 12.0, 200.0, 0.0, 0.0, 182.28, 0.0, 1.0]
 : My high_level_data is {'vehicle_rep': {'Knowledge Reporter 1': None, 'Knowledge Reporter 2': None}}
Learners Reporter 2: RENDER_REASONING
 : My low_level_data is [946707839.0, 30.0, 12.0, 200.0, 0.0, 0.0, 182.28, 0.0, 1.0]
 : My high_level_data is {'vehicle_rep': {'Knowledge Reporter 1': None, 'Knowledge Reporter 2': None}}
Planner Reporter 1: UPDATE
 : headers ['Time', 'VOLTAGE', 'CURRENT', 'THRUST', 'ALTITUDE', 'ACCELERATION', 'TEMPERATURE', 'SCIENCE_COLLECTION', 'LABEL_ERROR_STATE']
 : low_level_data <class 'list'> = '[]'
 : high_level_data <class 'dict'> = '{'vehicle_rep': {'Knowledge Reporter 1': None, 'Knowledge Reporter 2': None}, 'learning_systems': {'Learners Reporter 1': None, 'Learners Reporter 2': None}}'
Planner Reporter 2: UPDATE
 : headers ['Time', 'VOLTAGE', 'CURRENT', 'THRUST', 'ALTITUDE', 'ACCELERATION', 'TEMPERATURE', 'SCIENCE_COLLECTION', 'LABEL_ERROR_STATE']
 : low_level_data <class 'list'> = '[]'
 : high_level_data <class 'dict'> = '{'vehicle_rep': {'Knowledge Reporter 1': None, 'Knowledge Reporter 2': None}, 'learning_systems': {'Learners Reporter 1': None, 'Learners Reporter 2': None}}'
Planner Reporter 1: RENDER_REASONING
 : My low_level_data is []
 : My high_level_data is {'vehicle_rep': {'Knowledge Reporter 1': None, 'Knowledge Reporter 2': None}, 'learning_systems': {'Learners Reporter 1': None, 'Learners Reporter 2': None}}
Planner Reporter 2: RENDER_REASONING
 : My low_level_data is []
 : My high_level_data is {'vehicle_rep': {'Knowledge Reporter 1': None, 'Knowledge Reporter 2': None}, 'learning_systems': {'Learners Reporter 1': None, 'Learners Reporter 2': None}}
Complex Reporter 1: UPDATE
 : headers ['Time', 'VOLTAGE', 'CURRENT', 'THRUST', 'ALTITUDE', 'ACCELERATION', 'TEMPERATURE', 'SCIENCE_COLLECTION', 'LABEL_ERROR_STATE']
 : low_level_data <class 'list'> = '[]'
 : high_level_data <class 'dict'> = '{'vehicle_rep': {'Knowledge Reporter 1': None, 'Knowledge Reporter 2': None}, 'learning_systems': {'Learners Reporter 1': None, 'Learners Reporter 2': None}, 'planning_systems': {'Planner Reporter 1': None, 'Planner Reporter 2': None}, 'complex_systems': {}}'
Complex Reporter 1: RENDER_REASONING
 : My low_level_data is []
 : My high_level_data is {'vehicle_rep': {'Knowledge Reporter 1': None, 'Knowledge Reporter 2': None}, 'learning_systems': {'Learners Reporter 1': None, 'Learners Reporter 2': None}, 'planning_systems': {'Planner Reporter 1': None, 'Planner Reporter 2': None}, 'complex_systems': {}}
Complex Reporter 2: UPDATE
 : headers ['Time', 'VOLTAGE', 'CURRENT', 'THRUST', 'ALTITUDE', 'ACCELERATION', 'TEMPERATURE', 'SCIENCE_COLLECTION', 'LABEL_ERROR_STATE']
 : low_level_data <class 'list'> = '[]'
 : high_level_data <class 'dict'> = '{'vehicle_rep': {'Knowledge Reporter 1': None, 'Knowledge Reporter 2': None}, 'learning_systems': {'Learners Reporter 1': None, 'Learners Reporter 2': None}, 'planning_systems': {'Planner Reporter 1': None, 'Planner Reporter 2': None}, 'complex_systems': {'Complex Reporter 1': None}}'
Complex Reporter 2: RENDER_REASONING
 : My low_level_data is []
 : My high_level_data is {'vehicle_rep': {'Knowledge Reporter 1': None, 'Knowledge Reporter 2': None}, 'learning_systems': {'Learners Reporter 1': None, 'Learners Reporter 2': None}, 'planning_systems': {'Planner Reporter 1': None, 'Planner Reporter 2': None}, 'complex_systems': {'Complex Reporter 1': None}}

```
