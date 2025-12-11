from abc import ABC, abstractmethod
from ..boundary import BC
from ...expr import Expr

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Set, Union
from ..discretization import Discretization

class Field(ABC):
    def __init__(
        self,
        disc: Discretization
    ):
        super().__init__()
        self._disc = disc
    
    @property
    def disc(self) -> Discretization:
        return self._disc

    @property
    @abstractmethod
    def bcs(self) -> Set[BC]:
        pass

    @abstractmethod
    def apply_bc(self, boundary_condition: BC):
        pass

    @abstractmethod
    def __add__(self, other: Union[Field, Expr, float, int]) -> Expr:
        pass

    @abstractmethod
    def __radd__(self, other: Union[Field, Expr, float, int]) -> Expr:
        pass

    @abstractmethod
    def __sub__(self, other: Union[Field, Expr, float, int]) -> Expr:
        pass

    @abstractmethod
    def __rsub__(self, other: Union[Field, Expr, float, int]) -> Expr:
        pass

    @abstractmethod
    def __mul__(self, other: Union[Field, Expr, float, int]) -> Expr:
        pass

    @abstractmethod
    def __rmul__(self, other: Union[Field, Expr, float, int]) -> Expr:
        pass

    @abstractmethod
    def __truediv__(self, other: Union[Field, Expr, float, int]) -> Expr:
        pass

    @abstractmethod
    def __rtruediv__(self, other: Union[Field, Expr, float, int]) -> Expr:
        pass

