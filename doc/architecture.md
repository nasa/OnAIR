
# **WIP: Under Construction and not yet complete.**

The cognitive architecture is the heart of OnAIR and consists of several components.

- [Configuration Files](#the-configuration-files)
  - [Initialization File](#initialization-ini-file)
  - [Telemetry Definitions File](#telemetry-definitions-json-file)
- [OnAirDataSource](#the-onairdatasource)
- [Pipeline](#the-pipeline)
  - [Data Types](#data-types)
  - [Plugins](#the-plugins)

## The Configuration Files
These files define the components and data structure that will be used by a running instance of OnAIR.

### Initialization (.ini) File
Provides the details about the components and other files used by OnAIR to perform its data reasoning.

It consists of the following sections:

- **[FILES]** - Pathing and file name information, it is a required section and all following keys are required
  - TelemetryFilePath (string) - location of the telemetry file on the local system
  - TelemetryFile (string) - name of the telemtry file, used by file access [OnAir data sources](#the-onairdatasource)
  - MetaFilePath (string)- location of the telemtry definitions file
  - MetaFile (string) - name of the [telemetry definitions file](#telemetry-definitions-json-file), a JSON file

- **[DATA_HANDLING]** - The telemetry data parser, it is a required section and the following key is required
  - DataSourceFile (string) - full path and python file name of the [OnAir data source](#the-onairdatasource) to use

- **[PLUGINS]** - The various [plugins](#the-plugins) that OnAIR will use to construct the [pipeline](#the-pipeline), it is a required section and all following keys are required
  - KnowledgeRepPluginDict (dict)
  - LearnersPluginDict (dict)
  - PlannersPluginDict (dict)
  - ComplexPluginDict (dict)

  Each plugin dictionary consists of key, name for the plugin at runtime, to value, path to the plugin's base directory. The key name is not required to be the same as the name of the plugin, but the directory used for the plugin must contain the concrete plugin class named **\<directory name\>_plugin.py**. Multiple of the same plugin directory may be used by defining unique key names, e.g. {"Reporter1":"plugins/reporter", "Reporter2":"plugins/reporter"} will provide two reporter plugins. A plugin type may be left empty by giving it an empty dict, **{}**.

- **[OPTIONS]** - Provides for any optional settings, section and its key are optional
  - IO_Enabled (bool) - this determines whether or not to provide text output during runtime, omission of this equates to **_false_**

### Telemetry Definitions (.json) File
Also known as the 'Metafile', this provides the information necessary for OnAIR to understand the received data.

## The OnAirDataSource
This is the adapter that attaches OnAIR to a telemetry source. It provides the basic data that is used by the cognitive components to render reasoning about the system.

There are a few provided by default (located in the [onair/data_handling](https://github.com/nasa/OnAIR/tree/686df367bf4b9679ee9be11524230e99a499e5f0/onair/data_handling) directory):
- CSV - data from a .csv formatted file
- Redis - data from Redis server channels
- SBN - data via the Software Bus Network attached to an instance of [NASA's core Flight Software](https://github.com/nasa/cFS "cFS on GitHub")

However, all of these just implement the abstract class `OnAirDataSource` and users are able (nay, encouraged!) to write their own.

## The Pipeline
This is the path in which the received data from the source will flow.

The order of plugin operations is determined by placement in the plugin list within the [initilization](#initialization-ini-file) file used when running OnAIR. Plugins run from first to last in the order they are placed within their dictionary definitions.

### Data Types
There are two types of data used by the constructs in OnAIR, low and high level data

- `low_level_data`
  - also referred to as the "frame"
  - current snapshot of data received through whatever [OnAirDataSource](#the-onairdatasource) is in use

- `high_level_data`
  - reasoned data from plugins
  - built and grows as the [pipeline](#the-pipeline) is traversed


### The Plugins

There are two methods required to be implemented by OnAIR for the plugins, `update` and `render_reasoning`.
- `update` pulls data into the plugin for use, this includes either low_level_data, high_level_data, or both dependent upon the plugin type
- `render_reasoning` pushes out high_level_data in the form of a python dict with key/value pairs, i.e., this is the data you want made available to plugins further down the pipeline

What a plugin does with the data and how it renders its reasoning is up to the plugin developer to decide. However, the order of operations is established by the initialize file used and OnAIR.

### Plugin Flow

- Knowledge Representations (KR) - run first
  - all KRs run `update`, receiving ONLY `low_level_data`
  - all KRs run `render_reasoning`, returning their respective dictionaries of reasonings
  - all KR reasonings are put into the `high_level_data` as "\<KR plugin name\>:{\<returned reasoning dict\>}"
- Learners (LN) - run second
  - all KRs run `update`, receiving `low_level_data` and all KR's resonings in the `high_level_data`
  - LNs do not receive other LNs reasonings
  - all LNs run `render_resoning`, returning their respective dictionaries of reasonings
  - all LN resonings are put into the `high_level_data` as "\<LN plugin name\>:{\<returned reasoning dict\>}"
- Planners (PL) - run third
  - all PLs run `update`, receiving ONLY KR's and LN's reasonings in the `high_level_data`
  - PLs do not receive other PLs resonings
  - all PLs run `render_resoning`, returning their respective dictionaries of reasonings
  - all PL resonings are put into the `high_level_data` as "\<PL plugin name\>:{\<returned reasoning dict\>}"
- Complex Reasoners (CR) - run last
  - each CR runs `update` and `render_resoning` receiving all KR, LN, PL, and any previous CR's reasonings in the `high_level_data`
  - each CR completes resoning, returning their respective dictionaries of reasonings
  - each CR resonings are put into the `high_level_data` as "\<CR plugin name\>:{\<returned reasoning dict\>}"
  - CRs in turn get progressively more data than the one before it

Here is a illustration of this relationship:
[insert image here]


The best way to "see" this relationship is to run the reporter example:
```
python driver.py onair/config/reporter_config.ini
```
sample output:
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

This gives some minimal output; however, you can edit the reporter plugin ([plugin/reporter/reporter_plugin.py](https://github.com/nasa/OnAIR/blob/686df367bf4b9679ee9be11524230e99a499e5f0/plugins/reporter/reporter_plugin.py) line 14) and change verbose_mode to True:

```
...
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
...
```
