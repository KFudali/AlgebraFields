from typing import Self
import numpy as np

from .operator import Operator
from ..expression import Expression, ScalarExpression
from algebra.exceptions import ShapeMismatchException


class CombinedOperator(Operator):
    def __init__(self, Ax: Operator, b: Expression):
        if Ax.output_shape != b.output_shape:
            raise ShapeMismatchException((
                "Operator output shape must match Expression shape",
                f"Operator: {Ax.output_shape}, Expression: {b.output_shape}."
            ))
        super().__init__(Ax.input_shape, b.output_shape)
        self._Ax = Ax
        self._b = b

    def _apply(self, field: np.ndarray, out: np.ndarray):
        self._Ax.apply(field, out)
        out += self._b.eval()

    def __neg__(self) -> Self:
        return CombinedOperator(-self._Ax, - self._b)

    def __add__(self, other: float | Expression | ScalarExpression) -> Self:
        if isinstance(other, Operator):
            return CombinedOperator(self._Ax + other, self._b)
        if isinstance(other, (float, ScalarExpression)):
            return CombinedOperator(self._Ax, self._b + other)
        return NotImplemented

    def __mul__(self, other: float | ScalarExpression) -> Self:
        if isinstance(other, (float, ScalarExpression)):
            return CombinedOperator(self._Ax * other, self._b * other)
        return NotImplemented

    def __truediv__(self, other: float | ScalarExpression) -> Self:
        if isinstance(other, (float, ScalarExpression)):
            return CombinedOperator(self._Ax / other, self._b / other)
        return NotImplemented