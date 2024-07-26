# Using OnAIR with Core Flight System (cFS)

## Overview

In order for OnAIR to receive cFS messages, cFS must include the Software Bus Network (SBN) app and the SBN Client library in its tree.

SBN is a cFS application that enables pub/sub across the software busses (SB) of multiple cFS instances: https://github.com/nasa/SBN

SBN Client is an external library that implements the SBN protocol. This allows other environments, including Python, to communicate with a Software Bus via the SBN: https://github.com/nasa/SBN-Client

In OnAIR, the sbn_adapter DataSource uses SBN Client to subscribe to messages on the Software Bus. When cFS messages are received, it extracts the data from the message structs and passes it up to the rest of the OnAIR system.

## Example
An example distribution of cFS integrated with OnAIR can be found [here.](https://github.com/the-other-james/cFS/tree/OnAIR-integration)

In this example, OnAIR is configured to the subscribe to the sample_app house keeping telemetry packet, `SAMPLE_APP_HkTlm_t`.

### Quick Start
Requirements: Docker

After cloning the repository, use `docker compose` to build and run a container called `cfs-onair` which should have both cFS and OnAIR dependencies set up.

``` bash
docker compose up -d
```

After building, Docker will start the `cfs-onair` container in the background. Attach to the container using `docker exec`.

```bash
docker exec -it cfs-onair bash
```

Inside the `cfs-onair` container, use the build script to build cFS. SBN, SBN Client and OnAIR are added as cFS apps that will be built/installed by the cFS CMake system.

``` bash
./_build.sh
```

Then navigate to the install directory and run cFS.

``` bash
cd build/exe/cpu1
./core-cpu1
```

Open a new terminal. Then attach to the same `cfs-onair` container and navigate to the same intall directory. 

``` bash
docker exec -it cfs-onair bash
cd build/exe/cpu1
```

In this example, the build system copies OnAIR, the OnAIR telemetry metadata and config files, the ctypes structs of the cFS messages and the `sbn_python_client` python module from `sbn_client` files to `cf/`. The `sbn-client.so` binary is also installed to the `cf/` directory.

Now OnAIR can be run from the install directory `build/exe/cpu1`. 

```bash
python3 cf/onair/driver.py cf/onair/cfs_sample_app.ini
```

If OnAIR successfully connects to cFS via SBN Client and SBN, you should see the following start up log

```
SBN Adapter ignoring data file (telemetry should be live)
SBN Client library loaded: '<CDLL 'sbn_client.so', handle 11d7eef0 at 0xffff93b1f910>'
SBN_Client Connecting to 127.0.0.1, 2234

SBN Client init: 0
SBN Client command pipe: 0
SBN_Adapter Running
SBN Client subscribe msg (id 0x883): 0

***************************************************
************    SIMULATION STARTED     ************
***************************************************
App message received: MsgId 0x00000883
Payload: <message_headers.SAMPLE_APP_HkTlm_Payload_t object at 0xffff93ae7140>
```

### Explanation

This repo is essentially the base distribution of cFS with three additional `apps` added, `sbn`, `sbn_client` and `onair_app`. While SBN_Client and OnAIR can exist outside of cFS as standalone projects, they were added as cFS apps to take advantage of the cFS build system.

### SBN and SBN Client
SBN and SBN Client are added as submodules in the `apps/` folder. They are added to the build system by appending them to the global app list found in the `targets.cmake` file. Since SBN is an actual cFS app it also needs to be added to the cfe start up script, `cpu1-cfe_es_startup.scr`. When SBN Client is built, it results in `sbn-client.so` binary that other processes, such as OnAIR, will use to communicate with the SBN.

By default, SBN's configuration table (`sbn_conf_tbl.c`) will assign the address and port `127.0.0.1:2234` to this instance of the SBN app. In order for SBN_Client to talk to SBN it must use the same address, which is set in `sbn_client_defs.h`.

### OnAIR
An additional folder, called `onair_app/` is also added to the `apps/` folder. This folder contains the OnAIR source directory as a submodule, as well as a telemetry metadata file, config file, and a message_headers.py file. The `onair_app` also has a `CMakeLists.txt` file directing how it should be built by the cFS build system. In this case it just copies the files to the target directory.

`s_o.ini` - this is the config file. It specifies the name/location of the telemetry metadata file and selects `sbn_adapter` as the OnAIR Datasource.

`s_o_TLM_CONFIG.json` - this is the telemetry metadata file. It lists each data field that OnAIR is interested in as well as their order. Most importantly, in the `channels` field it matches the cFS message ID with name of the message and the name of the actual message struct. `sbn_adapter` will subscribe to the message IDs listed here. When a message is received `sbn_adapter` will use the received message IDs to determine the structure of the message so it can correctly unpack it.

```
    "channels":{
        "0x0883": ["SAMPLE_APP", "SAMPLE_APP_HkTlm_t"]
    }
```

`message_headers.py` - this python file is unqiue to `sbn_adapter.py` (i.e its not needed anywhere else in OnAIR). It defines the message structs that will be received using python ctypes. In this example `message_headers.py` contains the sample app house keeping message structs found in `sample_app/fsw/src/sample_app_msg.h`.


