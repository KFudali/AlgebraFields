import numpy as np
from typing import Self, Generic

from .operator import Operator, TOperator
from algebra.expression import Expression
from algebra.exceptions import ShapeMismatchException

class OperatorExpression(Expression, Generic[TOperator]):
    def __init__(self, input: Expression, operator: TOperator):
        super().__init__(operator.output_shape)
        self._input = input
        self._operator = operator

        if input.output_shape != operator.input_shape:
            raise ShapeMismatchException(
                f"""Output shape of input expression must match operator input shape.
Expression output_shape: {input.output_shape}
Operator input_shape: {operator.input_shape}
"""
            )

    @property
    def input(self) -> Expression:
        return self._input

    @property
    def operator(self) -> TOperator:
        return self._operator

    def eval(self) -> np.ndarray:
        out = np.zeros(shape=self.output_shape)
        input = self._input.eval()
        self._operator.apply(input, out)
        return out

    def copy(self):
        return OperatorExpression(self.input.copy(), self.operator.copy())

    def _new(self, input: Expression, operator: TOperator) -> Self:
        return OperatorExpression(input, operator)

    def __neg__(self):
        return self._new(-self._input, -self._operator)

    def __add__(self, other: Operator | Expression | float) -> Self:
        if isinstance(other, Operator):
            return self._new(self._input, self._operator + other)
        elif isinstance(other, Expression) or isinstance(other, float):
            return self._new(self._input + other, self._operator)

    def __mul__(self, other: Operator | Expression | float) -> Self:
        if isinstance(other, Operator):
            return self._new(self._input, self._operator * other)
        elif isinstance(other, Expression) or isinstance(other, float):
            return self._new(self._input * other, self._operator)

    def __truediv__(self, other: Operator | Expression | float) -> Self:
        if isinstance(other, Operator):
            return self._new(self._input, self._operator / other)
        elif isinstance(other, Expression) or isinstance(other, float):
            return self._new(self._input / other, self._operator)