from space.core import AbstractField
from algebra import Expression

class FieldUpdate():
    def __init__(
        self, 
        field: AbstractField, 
        expr: Expression,
    ):
        field.assert_array_shape(expr)
        self._field = field
        self._expr = expr

    def eval(self):
        self._field._set_current(self._expr.eval())