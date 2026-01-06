from typing import TYPE_CHECKING
import numpy as np

from tools.algebra import Expression, Operator, exceptions
if TYPE_CHECKING:
    from ..field import Field

class FieldOperator(Expression, Operator):
    def __init__(self, field: "Field", op: Operator):
        if field.space.shape != op.input_shape:
            raise exceptions.ShapeMismatchException(
                "Cannot create FieldOperator. Field size differs from operator size. "
                f"Field shape: {field.shape}, ",
                f"Operator input shape: {op.input_shape}"
            )
        if field.space.shape != op.output_shape:
            raise exceptions.ShapeMismatchException(
                "Cannot create FieldOperator. Field size differs from operator size. "
                f"Field shape: {field.shape}, ",
                f"Operator output shape: {op.input_shape}"
            )
        self._op = op
        self._field = field
        Expression.__init__(self, field.shape)
        Operator.__init__(self, field.shape, field.shape)

    def _apply(self, field: np.ndarray, out: np.ndarray):
        for component in range(self._field.components):
            self._op.apply(field, out[component:])

    def eval(self) -> np.ndarray:
        out = np.zeros(self._field.shape)
        for component in range(self._field.components):
            self._op.apply(self._field.value().eval(), out[component:])
        return out