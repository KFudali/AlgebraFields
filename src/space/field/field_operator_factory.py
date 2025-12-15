from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .field import Field

import expr

class FieldOperatorFactory():
    def __init__(self, field: Field):
        self._field = field

    def laplace(self):
        return expr.operators.LaplaceOperator(self._field)