from .field_expr import FieldExpr
from spacer.base import AbstractField
from .field_value import FieldValue

class FieldUpdate(FieldExpr):
    def __init__(
        self, 
        field: AbstractField, 
        value: FieldValue
    ):
        super().__init__(field.desc)
        assert field.same_shape(value)
        self.field = field
        self.value = value
        assert self.field.same_shape(value)

    def eval(self):
        self.field.raw_value.set(self.value.eval())