# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

"""
Create a persistent object to manage OnAIR services.
"""

from onair.src.util.singleton import Singleton
from onair.src.util.service_import import import_services


# pylint: disable=too-few-public-methods
class ServiceManager(Singleton):
    """
    A class to manage services.

    Attributes
    ----------
    services : dict
        A dictionary containing service names and their associated functions.
    """

    def __init__(self, service_dict=None):
        """
        Construct all service-related attributes on first instantiation.

        Parameters
        ----------
        service_dict : dict, optional
            Service names paired with their class file path.

        Raises
        ------
        ValueError
            If 'service_dict' parameter is not provided on first instantiation.

        Note
        ----
        Properly follows singleton behavior by only allowing initialize behavior to run
        upon first instantiation, subsequent initializations will ignore any provided service_dict.
        """
        if not hasattr(self, "services"):
            # Ensure service info is provided on the first instantiation
            if service_dict is None:
                raise ValueError(
                    "'service_dict' parameter required on first instantiation"
                )
            imported_services = import_services(service_dict)
            self.services = {}
            for service_name, service in imported_services.items():
                # Set self.<service_name> = service
                setattr(self, service_name, service)

                # Store service function name
                # self.services[service_name] = functions provided by that service
                service_funcs = set()
                for f in dir(service):
                    # Only add the f if it's a function and not "private"
                    if callable(getattr(service, f)) and not f.startswith("_"):
                        service_funcs.add(f)
                self.services[service_name] = service_funcs

    def get_services(self):
        """
        Return a dictionary of services and their functions.

        Returns
        -------
        dict
            A dictionary containing service names as keys and sets of their
            associated function names as values.
        """
        return self.services
