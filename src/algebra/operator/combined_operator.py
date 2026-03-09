from typing import Self, Optional
import numpy as np

from .operator import Operator
from ..expression import Expression, ScalarExpression, ZeroExpression
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

    @property
    def core(self) -> "CombinedOperator":
        return self

    @property
    def Ax(self) -> Operator:
        return self._Ax

    @property
    def b(self) -> Expression:
        return self._b

    def _apply(self, field: np.ndarray, out: np.ndarray):
        self.Ax.apply(field, out)
        out += self.b.eval()
        return out

    def copy(self):
        return CombinedOperator(self._Ax.copy(), self._b.copy())

    def take_b(self) -> Expression:
        b = self._b
        self._b = ZeroExpression(self.output_shape)
        return b

    def __neg__(self) -> Self:
        return CombinedOperator(-self._Ax, - self._b)

    def __add__(self, other: Operator | float | Expression | ScalarExpression) -> Self:
        if isinstance(other, CombinedOperator):
            return CombinedOperator(self._Ax + other._Ax, self._b + other._b)
        if isinstance(other, Operator):
            return CombinedOperator(self._Ax + other, self._b)
        if isinstance(other, (float, Expression, ScalarExpression)):
            return CombinedOperator(self._Ax, self._b + other)
        return NotImplemented

    def __sub__(self, other: Operator | float | Expression | ScalarExpression):
        return self + (-other)

    def __rsub__(self, other: Operator | float | Expression | ScalarExpression):
        return (-self) + other

    def __radd__(self, other: Operator | float | Expression | ScalarExpression) -> Self:
        if isinstance(other, CombinedOperator):
            return CombinedOperator(self._Ax + other._Ax, self._b + other._b)
        if isinstance(other, Operator):
            return CombinedOperator(self._Ax + other, self._b)
        if isinstance(other, (float, Expression, ScalarExpression)):
            return CombinedOperator(self._Ax, self._b + other)

    def __mul__(self, other: float | ScalarExpression) -> Self:
        if isinstance(other, (float, ScalarExpression)):
            return CombinedOperator(self._Ax * other, self._b * other)
        return NotImplemented

    def __rmul__(self, other: float | ScalarExpression) -> Self:
        if isinstance(other, (float, ScalarExpression)):
            return CombinedOperator(self._Ax * other, self._b * other)
        return NotImplemented

    def __truediv__(self, other: float | ScalarExpression) -> Self:
        if isinstance(other, (float, ScalarExpression)):
            return CombinedOperator(self._Ax / other, self._b / other)
        return NotImplemented