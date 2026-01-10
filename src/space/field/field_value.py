import numpy as np
from space.core import AbstractField, FieldExpression


class FieldValue(FieldExpression):
    def __init__(
        self, 
        field: AbstractField, 
        past_offset: int = 0
    ):
        super().__init__(field.space, field.components)
        self._field = field
        self._past_offset = past_offset

    def eval(self) -> np.ndarray:
        if self._past_offset == 0:
            return self._field._get_current()
        return self._field._get_past(self._past_offset)