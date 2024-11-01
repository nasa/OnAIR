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


def import_services(service_dict):
    services_dict = {}
    init_filename = "__init__.py"
    """
    services_dict format = {service1: service1_args, service2: service2_args}
    e.g.
    fleetinterface: {'path': 'service/redis_fleet_interface',
                    'agent_callsign': 'agent1',
                    'connection_timeout': '3'}
    """
    for service, kwargs in service_dict.items():
        true_path = kwargs.pop('path') # path no longer needed afterwards
        # Last directory name is the module name
        mod_name = os.path.basename(true_path) # redis_fleet_interface
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
        service_name = f"{mod_name}_service"
        service = __import__(f"{mod_name}.{service_name}", fromlist=[service_name])
        # add an instance of the module's was an OnAIR plugin
        services_dict[mod_name] = service.Service(**kwargs)
    return services_dict
