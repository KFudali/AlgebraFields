from ..field import FieldExpr
from space.base import AbstractField

class Operator(FieldExpr):
    def __init__(self, field: AbstractField):
        super().__init__(field.desc)
        self._field = field

    @property
    def field(self) -> AbstractField:
        return self._field