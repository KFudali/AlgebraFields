from abc import ABC, abstractmethod

from .discrete_operator import DiscreteOperator

class DiscreteOperatorsFactory(ABC):
    @abstractmethod
    def eye(self) -> DiscreteOperator: pass

    @abstractmethod
    def laplace(self) -> DiscreteOperator: pass
    
    # @abstractmethod
    # def gradient(self, field_values: np.ndarray, grad: np.ndarray): pass