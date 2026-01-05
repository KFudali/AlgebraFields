from abc import ABC, abstractmethod
from .bcs import DiscreteDirichletBC, DiscreteNeumannBC
from model.domain import Boundary

class DiscreteBCs(ABC):
    @abstractmethod
    def dirichlet(
        self, boundary: Boundary, value: float
    ) -> DiscreteDirichletBC: pass
    
    @abstractmethod
    def neumann(
        self, boundary: Boundary, value: float
    ) -> DiscreteNeumannBC: pass