import numpy as np

from model.domain import Boundary
from model.discrete.core.bcs import DiscreteBC

from space.core import AbstractField, Space
from algebra.expression import Expression

from .field_update import FieldUpdate
from .field_value import FieldValue
from .value_buffer import ValueBuffer


class Field(AbstractField):
    def __init__(self, space: Space, components: int):
        super().__init__(space, components)
        self._value = ValueBuffer(self.shape)
        self._bcs = dict[Boundary, DiscreteBC]()

    def _get_current(self) -> np.ndarray:
        return self._value.get() 

    def _set_current(self, value: np.ndarray): 
        self._value.set_current(value)

    def _get_past(self, past_offset: int = 1) -> np.ndarray:
        return self._value.get(past_offset)
    
    def _advance(self):
        self._value.advance()
    
    def prev_value(self, past_offset: int = 1) -> FieldValue:
        if past_offset > self._value.saved_steps:
            self._value.set_saved_steps(past_offset)
        return FieldValue(self, past_offset)

    def current_value(self) -> FieldValue:
        return FieldValue(self, past_offset=0)

    def update_current(self, expression: Expression) -> FieldUpdate:
        return FieldUpdate(self, expression)
    
    def apply_bc(self, bc: DiscreteBC):
        self._bcs[bc.boundary] = bc

    @property
    def bcs(self) -> list[DiscreteBC]:
        return self._bcs.values()