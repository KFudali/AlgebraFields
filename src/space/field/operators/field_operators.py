import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..field import Field

from .field_operator import FieldOperatorExpr
from space.core import CallableFieldOperator

class FieldOperators():
    def __init__(self, field: "Field"):
        self._field = field
        self._disc = field.space.discretization
    
    def laplace(self) -> FieldOperatorExpr:
        def component_laplace(field: np.ndarray, out: np.ndarray):
            for comp in range(self._field.components):
                self._disc.operators.laplace().apply(
                    field[comp, :], out[comp, :]
                )

        op = CallableFieldOperator(
            self._field.space, 
            self._field.components,
            component_laplace
        )
        
        return FieldOperatorExpr(
            field=self._field, op=op
        )