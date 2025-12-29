import numpy as np

from space.base import FieldDescriptor, AbstractField, bc, FieldValueBuffer
from model.discretization import Discretization

from ...expr.field import FieldValue
from .field_operator_factory import FieldOperatorFactory

class Field(AbstractField):
    def __init__(
        self, value_buffer: FieldValueBuffer
    ):
        super().__init__(value_buffer)
        self._operator_factory = FieldOperatorFactory(field=self)
        self._bcs = set[bc.BC]()

    @property
    def disc(self) -> Discretization:
        return self.space.disc

    @property
    def bcs(self) -> set[bc.BC]: return self._bcs

    @property
    def operator(self) -> FieldOperatorFactory:
        return self._operator_factory

    def value(self) -> FieldValue:
        return FieldValue(self.desc, self.raw_value.get)

    def apply_bc(self, boundary_condition: bc.BC):
        self._bcs.add(boundary_condition)