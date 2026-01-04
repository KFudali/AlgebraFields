from abc import abstractmethod, ABC
import numpy as np
from algebra import Operator

from model.domain import Boundary
class DiscreteBC(ABC):
    def __init__(self, boundary: Boundary):
        super().__init__()
        self._boundary = boundary

    @property
    def boundary(self) -> Boundary:
        return self._boundary

    @abstractmethod 
    def apply_to_les(self, A: Operator, b: np.ndarray): pass

class DiscreteDirichletBC(DiscreteBC): pass
class DiscreteNeumannBC(DiscreteBC): pass