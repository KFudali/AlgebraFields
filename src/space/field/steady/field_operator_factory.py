from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .field import Field

from ...expr import operators

class FieldOperatorFactory():
    def __init__(self, field: "Field"):
        self._field = field

    def laplace(self):
        return operators.LaplaceOperator(self._field)