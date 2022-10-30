from abc import ABC, abstractmethod

class GenericIntelligenceConstruct(ABC):
    @abstractmethod
    def __init__(self, headers):
        """
        Abstract superclass for data driven components: VAE, PPO, etc. Allows for easier modularity.
        """
        pass

    @abstractmethod
    def apriori_training(self, batch_data):
        """
        Given data, system should learn any priors necessary for realtime diagnosis.
        """
        pass
        
    @abstractmethod
    def update(self, frame):
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
