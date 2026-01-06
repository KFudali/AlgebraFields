from abc import ABC, abstractmethod
from . import DiscreteDirichletBC, DiscreteNeumannBC
from ..domain import Boundary

class DiscreteBCs(ABC):
    @abstractmethod
    def dirichlet(
        self, boundary: Boundary, value: float
    ) -> DiscreteDirichletBC: pass
    
    @abstractmethod
    def neumann(
        self, boundary: Boundary, value: float
    ) -> DiscreteNeumannBC: pass