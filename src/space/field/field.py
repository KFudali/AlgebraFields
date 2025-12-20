from abc import ABC
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..field_bound import FieldBound, FieldDescriptor
    from .field_operator_factory import FieldOperatorFactory

from .field_value import FieldValue
from discretization import Discretization
from ..boundary import BC
import numpy as np

class Field(FieldBound, ABC):
    def __init__(
        self, field_descriptor: FieldDescriptor
    ):
        super().__init__(field_descriptor)
        self._operator_factory = FieldOperatorFactory(field=self)
        self._bcs = set[BC]()
        self._value = np.ndarray(self.shape, dtype=float)

    @property
    def bcs(self) -> set[BC]: return self._bcs

    @property
    def discretization(self) -> Discretization:
        return self._space.discretization

    @property
    def operator(self) -> FieldOperatorFactory:
        return self._operator_factory

    def value(self) -> FieldValue:
        return FieldValue(self.desc, self._get_value)

    def _get_value(self) -> np.ndarray:
        return self._value

    def set_value(self, raw_value: np.ndarray) -> bool:
        if raw_value.shape !=  self.shape: return False
        else: self._value = raw_value

    def apply_bc(self, boundary_condition: BC):
        self._bcs.add(boundary_condition)