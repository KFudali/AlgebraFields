from tools.buffer import ValueBuffer, ShiftProxyValueBuffer

import algebra
from space.core import Space
from .core import AbstractField, FieldValue, FieldUpdate

class FieldView(AbstractField):
    def __init__(
        self, 
        space: Space, components: int,
        value_buffer: ValueBuffer
    ):
        super().__init__(space, components)
        if self.shape != self._value_buffer:
            raise algebra.exceptions.ShapeMismatchException(
                "Passed value buffer does not match field shape"
            )
        self._value_buffer = value_buffer

    def value(self) -> FieldValue:
        return FieldValue(self)

    def save_past(self, steps: int):
        self._value_buffer.set_saved_steps(steps)

    def past(self, steps: int = 1) -> "FieldView":
        self.save_past(steps)
        return FieldView(ShiftProxyValueBuffer(self._value_buffer, steps))

    def advance(self):
        self._value_buffer.advance()

class Field(FieldView):
    def __init__(
        self, 
        space: Space, components: int,
        value_buffer: ValueBuffer,
    ):
        super().__init__(space, components, value_buffer)

    def set_value(self, expr: algebra.Expression) -> FieldUpdate:
        return FieldUpdate(self, expr)