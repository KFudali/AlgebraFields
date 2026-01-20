from abc import ABC, abstractmethod 
import numpy as np
from space.core import AbstractField, FieldLinearOperator
from space.field import Field, FieldValue
from space.field.value_buffer import ValueBuffer

class FieldTimeDerivative(AbstractField, ABC):
    def __init__(self, field: Field, required_time_steps: int):
        super().__init__(field.space, field.components)
        self._value = ValueBuffer(field.shape)
        self._dts: list[float] = [1.0]
        self._source = field
        self._req_time_steps = required_time_steps
        field.set_saved_steps(required_time_steps)

    @property
    def field(self) -> Field: return self._source

    @property
    def required_time_steps(self) -> int:
        return self._req_time_steps

    def prev_value(self, past_offset: int = 1) -> FieldValue:
        return FieldValue(self, past_offset)

    def value(self) -> FieldValue:
        return FieldValue(self, past_offset=0)

    def _set_current(self, value: np.ndarray):
        self._set_current(value)

    def _get_current(self) -> np.ndarray:
        return self._value.get()

    def _get_past(self, past_step: int = 1) -> np.ndarray:
        return self._value.get(past_step)

    def advance(self, dt: float):
        self._advance(dt)

    def _dt(self, step: int = 0) -> float:
        return self._dts[step]

    def _advance(self, dt: float):
        self._value.advance()
        self._dts.append(dt)
        if len(self._dts) > 1 + self.required_time_steps:
            self._dts.pop(-1)

    @abstractmethod
    def _calculate(self, dt: float) -> np.ndarray: pass
    
    @abstractmethod
    def op(self) -> FieldLinearOperator: pass