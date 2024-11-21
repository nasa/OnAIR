# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

"""
This module defines the AIPlugIn abstract base class for the OnAIR Platform.

The AIPlugIn class serves as a foundation for all AI plugins in the OnAIR system.
It establishes standards and structures for compliance that must be adhered to by
the four imported plugin types: Knowledge Representation, Learner, Planner, or
Complex Reasoner.

This abstract base class defines the basic structure and required methods that
all AI plugins must implement.
"""


from abc import ABC, abstractmethod
from typing import Any


class AIPlugin(ABC):
    """
    This serves as a base for all plugins.

    This is the superclass for data driven components. It is
    meant to induce standards and structures of compliance for imported
    plugins: Knowledge Representation, Learner, Planner or Complex Reasoner.
    Those plugins must inherit this class.

    Attributes
    ----------
    component_name : str
        Name given to instance of plugin.
    headers : list of str
        Names for each data point in the OnAIR data frame.
    """

    def __init__(self, _name: str, _headers: list):
        """
        Initialize a new AIPlugIn object.

        Parameters
        ----------
        _name : str
            The name of this plugin instance.
        _headers : list of str
            Sequenced names of each item in OnAIR data frame.
        """
        assert len(_headers) > 0
        self.component_name = _name
        self.headers = _headers

    @abstractmethod
    def update(
        self,
        low_level_data: list = None,
        high_level_data: dict = None,
    ) -> None:
        """
        Update the plugin's internal state with new data.

        This method is called to update the plugin with the latest data from the system.
        It can process either low-level sensor data or high-level reasoning results
        from other plugins, or both dependent upon plugin type.

        Parameters
        ----------
        low_level_data : list of float, optional
            Raw sensor data as a list of floats, corresponding to the headers defined in the plugin.
        high_level_data : dict of {str: dict of {str: any}}, optional
            Processed data and reasoning results from other plugins, organized by plugin type and name.

        Returns
        -------
        None
            This method does not return any value.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def render_reasoning(self) -> Any:
        """
        Plugin reasons with current data and provides analysis.

        Returns
        -------
        Any
            The reasoned outcome, which can be of any type.
            May return None if there are no results.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError
