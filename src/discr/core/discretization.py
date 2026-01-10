from abc import ABC, abstractmethod
from typing import TypeVar, Generic
import numpy as np

from .discrete_operators import DiscreteOperators
from .bcs.discrete_bcs import DiscreteBCs
from .domain import Domain

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

    @abstractmethod
    def flatten(self, field_array: np.ndarray) -> np.ndarray: pass

    @abstractmethod
    def reshape(self, field_array: np.ndarray) -> np.ndarray: pass