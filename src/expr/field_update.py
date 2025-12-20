import space.field

from .field_expr import FieldExpr
from space.field.field_value import FieldValue

class FieldUpdate(FieldExpr):
    def __init__(self, field: space.field.Field, value: FieldValue):
        super().__init__()
        self.field = field
        self.value = value

    def eval(self):
        self.field.set_value(self.value.eval())