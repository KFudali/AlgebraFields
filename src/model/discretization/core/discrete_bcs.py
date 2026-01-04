from abc import ABC, abstractmethod
from .discrete_bc import DiscreteDirichletBC, DiscreteNeumannBC
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