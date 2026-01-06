from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..field import Field

from .field_operator import FieldOperator

class FieldOperators():
    def __init__(self, field: "Field"):
        self._field = field
        self._disc = field.space.discretization

    def laplace(self) -> FieldOperator:
        return FieldOperator(
            field=self._field, op=self._disc.operators.laplace()
        )