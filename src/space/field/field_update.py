from space.core import AbstractField
from tools.algebra import Expression

class FieldUpdate():
    def __init__(
        self, 
        field: AbstractField, 
        expr: Expression,
    ):
        self._field = field
        self._expr = expr

    def eval(self):
        self._field._set_current(self._expr.eval())