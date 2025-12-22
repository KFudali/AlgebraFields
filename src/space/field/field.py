import numpy as np

from space.base import FieldDescriptor, AbstractField, bc
from model.discretization import Discretization

from ..expr.field import FieldValue
from .field_operator_factory import FieldOperatorFactory

class Field(AbstractField):
    def __init__(
        self, field_descriptor: FieldDescriptor
    ):
        super().__init__(field_descriptor)
        self._operator_factory = FieldOperatorFactory(field=self)
        self._bcs = set[bc.BC]()
        self._value = self.disc.zeros()

    @property
    def disc(self) -> Discretization:
        return self.space.disc

    @property
    def bcs(self) -> set[bc.BC]: return self._bcs

    @property
    def operator(self) -> FieldOperatorFactory:
        return self._operator_factory

    def value(self) -> FieldValue:
        return FieldValue(self.desc, self.raw_value)

    def apply_bc(self, boundary_condition: bc.BC):
        self._bcs.add(boundary_condition)

    def raw_value(self) -> np.ndarray:
        return self._value
    
    def set_raw_value(self, raw_value: np.ndarray):
        assert raw_value.shape == self._value.shape
        self._value = raw_value