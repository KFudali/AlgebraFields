from abc import ABC, abstractmethod
from typing import TypeVar, Generic


import algebra
from .domain import TBoundary

class DiscreteBC(ABC, Generic[TBoundary]):
    def __init__(self, boundary: TBoundary):
        super().__init__()
        self._boundary = boundary

    @property
    def boundary(self) -> TBoundary:
        return self._boundary
    
    @abstractmethod
    def apply(self, operator: algebra.Operator) -> tuple[algebra.Operator, algebra.Expression]:
        pass


TDiscreteBC = TypeVar("TDiscreteBC", bound=DiscreteBC)