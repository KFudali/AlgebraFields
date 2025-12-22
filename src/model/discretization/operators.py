from abc import ABC, abstractmethod
import numpy as np

class DiscreteOperators(ABC):
    @abstractmethod
    def laplace(self, field_values: np.ndarray, laplace: np.ndarray): pass
    
    # @abstractmethod
    # def gradient(self, field_values: np.ndarray, grad: np.ndarray): pass