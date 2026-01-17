import numpy as np

from discr.core.domain import Boundary
from discr.core.bcs import DiscreteBC

from tools.algebra import Expression
from space.core import AbstractField, Space

from .field_update import FieldUpdate
from .field_value import FieldValue
from .value_buffer import ValueBuffer
from .operators import FieldOperators

class Field(AbstractField):
    def __init__(self, space: Space, components: int):
        super().__init__(space, components)
        self._value = ValueBuffer(self.shape)
        self._bcs = dict[Boundary, DiscreteBC]()
        self._ops = FieldOperators(field=self)

    def _get_current(self) -> np.ndarray:
        return self._value.get() 

    def _set_current(self, value: np.ndarray): 
        self._value.set_current(value)

    def _get_past(self, past_offset: int = 1) -> np.ndarray:
        return self._value.get(past_offset)

    def advance(self, dt: float):
        self._advance(dt)

    def _advance(self, dt: float):
        self._value.advance()

    def prev_value(self, past_offset: int = 1) -> FieldValue:
        if past_offset > self._value.saved_steps:
            self._value.set_saved_steps(past_offset)
        return FieldValue(self, past_offset)

    def value(self) -> FieldValue:
        return FieldValue(self, past_offset=0)

    def update(self, expression: Expression) -> FieldUpdate:
        return FieldUpdate(self, expression)

    def set_saved_steps(self, steps: int):
        self._value.set_saved_steps(steps)

    def apply_bc(self, bc: DiscreteBC):
        self._bcs[bc.boundary] = bc

    @property
    def bcs(self) -> list[DiscreteBC]:
        return self._bcs.values()
    
    @property
    def operator(self) -> FieldOperators:
        return self._ops