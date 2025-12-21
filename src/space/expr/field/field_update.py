from .field_expr import FieldExpr
from space.base import AbstractField
from .field_value import FieldValue

class FieldUpdate(FieldExpr):
    def __init__(
        self, 
        field: AbstractField, 
        value: FieldValue
    ):
        super().__init__()
        self.field = field
        self.value = value
        assert self.field.same_shape(value)

    def eval(self):
        self.field.set_raw_value(self.value.eval())