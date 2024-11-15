Using the redis_adapter.py as the DataSource, telemetry can be received through multiple Redis channels and inserted into the full data frame.

## OnAIR config file (.ini)

The redis_example.ini uses a very basic setup:
 - meta : redis_example_CONFIG.json
 - parser : onair/data_handling/redis_adapter.py
 - plugins : one of each type, all of them 'generic' that do nothing

## Telemetry config file (.json)

The telemetry file defines the subscribed channels, data frame and subsystems.
 - subscriptions : defines channel names where telemetry will be received
 - order : designation of where each 'channel.telemetry_item' is to be put in full data frame (the data header)
 - subsystems : data for each specific telemetry item (descriptions only for redis example)


## Receipt of telemetry

The Redis adapter expects any published telemetry on a channel to include:
 - time
 - every telemetry_item as described under "order" as 'channel.telemetry_item'

All messages sent must be json format (key to value) and will warn when it is not then discard the message (outputting what was received first). Keys should match the required telemetry_item names with the addition of "time." Values should be floats.

## Running the example

Start a Redis server on 'localhost', port:6379 (typical defaults)
```
redis-server
```

Start a second Redis server on 'localhost', port:6380
```
redis-server --port 6380
```

Start up OnAIR with the redis_example.ini file:
```
python driver.py onair/config/redis_example.ini
```
You should see:
```
---- Redis adapter connecting to server...

---- ... connected to server # 0!

---- Subscribing to channel: state_0 on server # 0

---- Redis adapter: channel 'state_0' received message type: subscribe.

---- ... connected to server # 1!

---- Subscribing to channel: state_1 on server # 1

---- Subscribing to channel: state_2 on server # 1

---- Redis adapter: channel 'state_1' received message type: subscribe.

---- Redis adapter: channel 'state_2' received message type: subscribe.

***************************************************
************    SIMULATION STARTED     ************
***************************************************
```

In another process run the experimental publisher:
```
python redis-experiment-publisher-multi-server.py
```
This will send telemetry every 2 seconds, one channel at random until all 3 channels have recieved data then repeat for a total of 9 times (all of which can be changed in the file). Its output should be similar (but randomly different) to this:
```
Published data to state_0, [0, 0.1, 0.2]
Published data to state_1, [1, 1.1, 1.2]
Published data to state_2, [2, 2.1, 2.2]
Completed 1 loops
Published data to state_2, [3, 3.1, 3.2]
Published data to state_1, [4, 4.1, 4.2]
```
And OnAir should begin receiving data similarly to this:
```
--------------------- STEP 1 ---------------------

CURRENT DATA: [0, 0.1, 0.2, '-', '-', '-', '-']
INTERPRETED SYSTEM STATUS: ---

--------------------- STEP 2 ---------------------

CURRENT DATA: [1, 0.1, 0.2, 1.1, 1.2, '-', '-']
INTERPRETED SYSTEM STATUS: ---

--------------------- STEP 3 ---------------------

CURRENT DATA: [2, 0.1, 0.2, 1.1, 1.2, 2.1, 2.2]
INTERPRETED SYSTEM STATUS: ---

--------------------- STEP 4 ---------------------

CURRENT DATA: [3, 0.1, 0.2, 1.1, 1.2, 3.1, 3.2]
INTERPRETED SYSTEM STATUS: ---

--------------------- STEP 5 ---------------------

CURRENT DATA: [4, 0.1, 0.2, 4.1, 4.2, 3.1, 3.2]
INTERPRETED SYSTEM STATUS: ---
...
```
