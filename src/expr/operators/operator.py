from ..field_expr import FieldExpr
import space.field


class Operator(FieldExpr):
    def __init__(self, field: space.field.Field):
        super().__init__()
        self._field = field

    @property
    def field(self) -> space.field.Field:
        return self._field