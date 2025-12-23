from abc import abstractmethod, ABC
import numpy as np
from model.domain import Boundary

class DiscreteBC(ABC):
    @abstractmethod 
    def apply_linear(
        self, A: np.ndarray, b: np.ndarray, value: float, boundary: Boundary
    ): pass

class DiscreteDirichletBC(DiscreteBC): pass
class DiscreteNeumannBC(DiscreteBC): pass