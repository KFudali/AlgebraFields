from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import numpy as np
if TYPE_CHECKING:
    from ..space_bound import SpaceBound, Space
    from .field_operator_factory import FieldOperatorFactory

from discretization import Discretization
from ..boundary import BC
import expr

class Field(SpaceBound, ABC):
    def __init__(
        self,
        space: Space
    ):
        SpaceBound.__init__(space = space)
        self._operator_factory = FieldOperatorFactory(field=self)
        self._bcs = set[BC]()

    @property
    def bcs(self) -> set[BC]: return self._bcs

    @property
    def discretization(self) -> Discretization:
        return self._space.discretization

    @property
    def operator(self) -> FieldOperatorFactory:
        return self._operator_factory

    def value(self) -> expr.FieldValue:
        return expr.FieldValue()

    def apply_bc(self, boundary_condition: BC):
        self._bcs.add(boundary_condition)