from __future__ import annotations

from typing import Callable, Self


class ScalarExpression:
    def __init__(self, expr: Callable[[], float]):
        self._expr = expr

    def eval(self) -> float:
        return self._expr()

    @staticmethod
    def ensure(value: float | "ScalarExpression") -> "ScalarExpression":
        if isinstance(value, ScalarExpression):
            return value
        return ScalarExpression(lambda: value)

    def __neg__(self) -> Self:
        return ScalarExpression(lambda: -self.eval())

    def __add__(self, other: ScalarExpression | float) -> Self:
        other = ScalarExpression.ensure(other)
        return ScalarExpression(lambda: self.eval() + other.eval())

    def __radd__(self, other: float) -> Self:
        return self + other

    def __sub__(self, other: ScalarExpression | float) -> Self:
        return self + (-ScalarExpression.ensure(other))

    def __rsub__(self, other: float) -> Self:
        return ScalarExpression.ensure(other) + (-self)

    def __mul__(self, other: ScalarExpression | float) -> Self:
        other = ScalarExpression.ensure(other)
        return ScalarExpression(lambda: self.eval() * other.eval())

    def __rmul__(self, other: float) -> Self:
        return self * other

    def __truediv__(self, other: ScalarExpression | float) -> Self:
        other = ScalarExpression.ensure(other)
        return ScalarExpression(lambda: self.eval() / other.eval())