from abc import ABC, abstractmethod
from .bc import DiscreteDirichletBC, DiscreteNeumannBC

class DiscreteBCs(ABC):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def dirichlet(self) -> DiscreteDirichletBC: pass
    
    @property
    @abstractmethod
    def neumann(self) -> DiscreteNeumannBC: pass