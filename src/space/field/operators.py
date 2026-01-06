from tools.algebra.expr import ApplyOperatorExpr
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..field import Field

class FieldOperators():
    def __init__(self, field: Field):
        self._field = field

    def laplace(self) -> ApplyOperatorExpr:
        laplace = self._field.space.discretization.operators.laplace()
        return ApplyOperatorExpr(laplace, self._field.current_value())