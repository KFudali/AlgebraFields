from space.core import Expression, AbstractField, time

class FieldUpdate():
    def __init__(
        self, 
        field: AbstractField, 
        expr: Expression,
        window: time.TimeWindow | None = None
    ):
        field.assert_array_shape(expr)
        self._field = field
        self._expr = expr
        self._window = window

    def _eval(self):
        self._field._set_current(self._window, self._expr.eval())