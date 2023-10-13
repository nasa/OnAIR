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

def import_plugins(headers, module_dict):
    plugin_list = []
    for module_name in list(module_dict.keys()):
                spec = importlib.util.spec_from_file_location(module_name, module_dict[module_name])
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                plugin_list.append(module.Plugin(module_name,headers))
    return(plugin_list)