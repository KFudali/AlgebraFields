from abc import ABC, abstractmethod
from typing import TypeVar, Generic
import numpy as np


from .discrete_bc_factory import DiscreteBCFactory
from .discrete_operator_factory import DiscreteOperatorsFactory
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
    def operators(self) -> DiscreteOperatorsFactory: pass

    @property
    @abstractmethod
    def bcs(self) -> DiscreteBCFactory: pass

    @abstractmethod
    def flatten(self, field_array: np.ndarray) -> np.ndarray: pass

    @abstractmethod
    def reshape(self, field_array: np.ndarray) -> np.ndarray: pass