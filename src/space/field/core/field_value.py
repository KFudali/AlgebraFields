import numpy as np
from .abstract_field import AbstractField
from .fieldshaped import FieldShaped
from algebra.expression import SimpleExpression

class FieldValue(FieldShaped, SimpleExpression):
    def __init__(self, field: AbstractField):
        super().__init__(field.space, field.components)
        self._field = field

    def eval(self) -> np.ndarray:
        return self._field.get_current()