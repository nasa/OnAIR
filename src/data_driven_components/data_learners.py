from abc import ABC, abstractmethod

class DataLearner(ABC):
    @abstractmethod
    def __init__(self):
        """
        Abstract superclass for data driven components: VAE, PPO, etc. Allows for easier modularity.
        """
        pass

    @abstractmethod
    def apriori_training(self):
        """
        Given data, system should learn any priors necessary for realtime diagnosis.
        """
        pass
    @abstractmethod
    def update(self):
        """
        Given streamed data point, system should update internally
        """
        pass
    @abstractmethod
    def render_diagnosis(self):
        """
        System should return its diagnosis
        """
        pass
