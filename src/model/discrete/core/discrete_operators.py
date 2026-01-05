from abc import ABC, abstractmethod

from algebra.operator import Operator

class DiscreteOperators(ABC):
    @abstractmethod
    def laplace(self) -> Operator: pass
    
    # @abstractmethod
    # def gradient(self, field_values: np.ndarray, grad: np.ndarray): pass