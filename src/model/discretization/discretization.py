from abc import ABC, abstractmethod
import numpy as np
from typing import TypeVar, Generic
from .stencils import Stencils
from .operators import DiscreteOperators
import model.domain

DomainType = TypeVar("DomainType", bound=model.domain.Domain)

class Discretization(ABC, Generic[DomainType]):
    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]: pass

    @abstractmethod
    def zeros(self) -> np.ndarray: pass

    @property
    @abstractmethod
    def domain(self) -> DomainType: pass
    
    @property
    @abstractmethod
    def dim(self) -> int: pass
    
    @property
    @abstractmethod
    def operators(self) -> DiscreteOperators: pass

    @property
    @abstractmethod
    def stencils(self) -> Stencils: pass