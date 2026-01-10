from typing import TYPE_CHECKING
import numpy as np

from space.core import FieldExpression, FieldOperator
from tools.algebra.exceptions import ShapeMismatchException
if TYPE_CHECKING:
    from ..field import Field

class FieldOperatorExpr(FieldExpression, FieldOperator):
    def __init__(self, field: "Field", op: FieldOperator):
        if field.shape != op.input_shape:
            raise ShapeMismatchException(
                "Cannot create FieldOperator. Field size differs from operator size. "
                f"Field shape: {field.shape}, ",
                f"Operator input shape: {op.input_shape}"
            )
        if field.shape != op.output_shape:
            raise ShapeMismatchException(
                "Cannot create FieldOperator. Field size differs from operator size. "
                f"Field shape: {field.shape}, ",
                f"Operator output shape: {op.input_shape}"
            )
        self._op = op
        self._field = field
        FieldExpression.__init__(self, field.space, field.components)
        FieldOperator.__init__(self, field.space, field.components)

    def _apply(self, field: np.ndarray, out: np.ndarray):
        return self._op._apply(field, out)
    
    def eval(self) -> np.ndarray:
        field = self._field.value().eval()
        out = np.zeros_like(field)
        self._apply(field, out)
        return out