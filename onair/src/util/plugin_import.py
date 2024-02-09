# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

"""
plugin_import.py
Function to import user-specified plugins (from config files) into interfaces
"""

import importlib.util
import sys
import os

def import_plugins(headers, module_dict):
    plugin_list = []
    init_filename = "__init__.py"
    for construct_name, module_path in module_dict.items():
        true_path = module_path
        # Compatibility for plugin paths that already include __init__.py
        if module_path.endswith(init_filename):
            true_path = module_path[:-len(init_filename) - 1]
        # Last directory name is the module name
        mod_name = os.path.basename(true_path)
        # import module if not already available
        if mod_name not in sys.modules:
            # add init file to get proper path for spec
            full_path = os.path.join(true_path, init_filename)
            # define spec for module loading
            spec = importlib.util.spec_from_file_location(mod_name, full_path)
            # create uninitialize module from spec
            module = importlib.util.module_from_spec(spec)
            # initialize the created module
            spec.loader.exec_module(module)
            # add plugin module to system for importation
            sys.modules[mod_name] = module
        # import the created module's plugin file for use
        plugin_name = f'{mod_name}_plugin'
        plugin = __import__(f'{mod_name}.{plugin_name}',
                            fromlist=[plugin_name])
        # add an instance of the module's was an OnAIR plugin
        plugin_list.append(plugin.Plugin(construct_name, headers))
    return(plugin_list)
