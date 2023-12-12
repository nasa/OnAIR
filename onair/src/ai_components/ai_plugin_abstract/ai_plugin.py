# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"


from abc import ABC, abstractmethod


class AIPlugIn(ABC):
    """This serves as a base for all plugins.

    This is the superclass for data driven components: VAE, PPO, etc. It is
    meant to induce standards and structures of compliance for imported
    plugins: KnowledgeRep, Learner, Planner or Complex. Those plugins must
    inherit this class.

    Attributes:
        component_name (str): Name given to instance of plugin.
        headers (list[str]): Names for each data point in a data frame.

    """
    def __init__(self, _name: str, _headers: list[str]):
        """Initializes a new AIPlugIn object.

        Args:
            _name: The name of this plugin instance.
            _headers: Sequenced names of each item in OnAIR data frame.

        """
        assert len(_headers) > 0
        self.component_name = _name
        self.headers = _headers

    @abstractmethod
    def update(
        self, low_level_data: list[float] = [],
        high_level_data: dict[str, dict[str, list[object]]] = {}
    ) -> None:
        """Updates plugin's data using provided data.

        Args:
            low_level_data: Frame of floats, one for each header.
            high_level_data: Reasoned data results from previous plugins.

        Returns:
        None: This function only updates the data for this plugin
        """
        raise NotImplementedError

    @abstractmethod
    def render_reasoning(self) -> list[object]:
        """Plugin reasons with current data and provides analysis.

        Returns:
            list[object]: The list of reasoned outcomes, where contents are
            relevant to this plugin. May be an empty list.
        """
        raise NotImplementedError
