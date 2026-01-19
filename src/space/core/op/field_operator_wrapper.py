import numpy as np
from .field_operator import FieldOperator
from tools.algebra.operator import Operator
from tools.algebra.exceptions import ShapeMismatchException
from space.core import Space

class FieldOperatorWrapper(FieldOperator):
    def __init__(
        self, space: Space, components: int, op: Operator
    ):
        super().__init__(space, components)
        if op.output_shape != self.output_shape:
            raise ShapeMismatchException(
                "Can only wrap operator that matches field shape",
                f"Field shape: {self.shape}",
                f"Operator output_shape: {op.output_shape}"
            )
        if op.input_shape != self.input_shape:
            raise ShapeMismatchException(
                "Can only wrap operator that matches field shape",
                f"Field shape: {self.shape}",
                f"Operator input_shape: {op.input_shape}"
            )
        self._op = op

    def _apply(self, field: np.ndarray, out: np.ndarray):
        self._op.apply(field, out)