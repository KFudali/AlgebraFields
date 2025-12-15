from ..expr import Expr
import space.field


class Operator(Expr):
    def __init__(self, field: space.field.Field):
        super().__init__()
        self._field = field