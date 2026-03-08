from abc import ABC, abstractmethod
from typing import Generic

from .discrete_bc import DiscreteBC
from .domain import TBoundary

class DiscreteBCFactory(ABC, Generic[TBoundary]):
    @abstractmethod
    def dirichlet(self, boundary: TBoundary, value: float) -> DiscreteBC: pass
    
    @abstractmethod
    def neumann(self, boundary: TBoundary, value: float) -> DiscreteBC: pass