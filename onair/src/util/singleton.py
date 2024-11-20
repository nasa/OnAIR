# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

"""
Implement singleton design pattern.
"""


# pylint: disable=too-few-public-methods
class Singleton:
    """
    A class for implementing singleton design pattern.
    """

    def __new__(cls, *args):
        """
        If the class has never been created, create a new one.
        """
        if not hasattr(cls, "instance"):
            # Create the one and only instance of this class
            cls.instance = super(Singleton, cls).__new__(cls)
        return cls.instance
