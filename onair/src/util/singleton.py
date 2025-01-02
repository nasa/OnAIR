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

    This class ensures that only one instance of the class is created.

    Attributes
    ----------
    instance : cls
        The single instance of the Singleton subclass.
    """

    def __new__(cls, *args, **kwargs):
        """
        Ensures only one instance of the cls is created.

        This method overrides the default __new__ to implement the Singleton pattern.
        It creates a new instance only if one doesn't already exist, otherwise
        it returns the existing instance.

        Parameters
        ----------
        cls : type
            The class that is instantiated, it must be a subclass of Singleton.
        *args : tuple
            Variable length argument list. Not used by Singleton, but may
            be used by the cls.
        **kwargs : dict
            Arbitrary keyword arguments. Not used by Singleton, but may
            be used by the cls.

        Returns
        -------
        instance : cls
            The single instance of the Singleton subclass.

        Note
        ----
        The __init__ method of the subclass will be called on every instantiation,
        therefore the subclass __init__ still needs to ensure proper singleton behavior.
        """
        if not hasattr(cls, "instance"):
            # Create the one and only instance of this class
            cls.instance = super(Singleton, cls).__new__(cls)
        return cls.instance
