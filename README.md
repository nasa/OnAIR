# RAISR-2.0

## Installing RAISR

Install headers, if on macOS:

    brew install cairo pkg-config
    brew install pygobject3 gtk+3

See [here](https://pycairo.readthedocs.io/en/latest/getting_started.html) and [here](https://pygobject.readthedocs.io/en/latest/getting_started.html#getting-started) for other systems.

You might run into libffy issues, see [here](https://github.com/mesonbuild/meson/issues/2273#issuecomment-420412230) and run (on macOS):

    export PKG_CONFIG_PATH="/usr/local/opt/libffi/lib/pkgconfig"

Create and activate a new conda environment (optional):

    conda create --name raisr2.0 python=3.9.5
    conda activate raisr2.0

Install requirements

    pip install -r pip.requirements.txt

If you run into

    AttributeError: module 'keras.utils.generic_utils' has no attribute 'populate_dict_with_module_objects'

Uninstall and reinstall latest tensorflow version

    pip uninstall tensorflow
    pip install tensorflow --upgrade --force-reinstall

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

