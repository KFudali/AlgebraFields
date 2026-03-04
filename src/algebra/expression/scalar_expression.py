from __future__ import annotations
from typing import Self, Callable

class ScalarExpression():
    def __init__(self, expr: Callable[[], float]):
        self._expr = expr
    
    def eval(self) -> float:
        return self._expr()

    def __neg__(self) -> Self:
        return ScalarExpression(lambda: -self.eval())

    def __add__(self, other: ScalarExpression | float) -> Self:
        if isinstance(other, ScalarExpression):
            return ScalarExpression(lambda: self.eval() + other.eval())
        elif isinstance(other, float):
            return ScalarExpression(lambda: self.eval() + other)

    def __mul__(self, other: ScalarExpression | float) -> Self:
        if isinstance(other, ScalarExpression):
            return ScalarExpression(lambda: self.eval() * other.eval())
        elif isinstance(other, float):
            return ScalarExpression(lambda: self.eval() * other)

    def __truediv__(self, other: ScalarExpression | float) -> Self:
        if isinstance(other, ScalarExpression):
            return ScalarExpression(lambda: self.eval() * other.eval())
        elif isinstance(other, float):
            return ScalarExpression(lambda: self.eval() * other)

    def __radd__(self, other: float) -> Self:
        return self + other

    def __sub__(self, other: Expression | float) -> Self:
        return self + (-other)

    def __rsub__(self, other: float) -> Self:
        return (-self) + other

    def __rmul__(self, other: float) -> Self:
        return self * other