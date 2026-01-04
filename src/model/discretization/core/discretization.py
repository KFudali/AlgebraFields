from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from .discrete_operators import DiscreteOperators
from .discrete_bcs import DiscreteBCs
from model.domain import Domain

DomainType = TypeVar("DomainType", bound=Domain)

class Discretization(ABC, Generic[DomainType]):
    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]: pass

    @property
    @abstractmethod
    def domain(self) -> DomainType: pass
    
    @property
    @abstractmethod
    def operators(self) -> DiscreteOperators: pass

    @property
    @abstractmethod
    def bcs(self) -> DiscreteBCs: pass