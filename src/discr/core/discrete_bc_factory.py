from abc import ABC, abstractmethod
from typing import Generic

from .discrete_bc import DiscreteBC
from .domain import BoundaryId

class DiscreteBCFactory(ABC):
    @abstractmethod
    def dirichlet(self, boundary: BoundaryId, value: float) -> DiscreteBC: pass
    
    @abstractmethod
    def neumann(self, boundary: BoundaryId, value: float) -> DiscreteBC: pass