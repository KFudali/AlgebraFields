from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable
import numpy as np

from algebra.exceptions import ShapeMismatchException
from ..expression import Expression
from .operators_mixin import ExpressionOperatorsMixin

class SimpleExpression(Expression, ExpressionOperatorsMixin):
    @abstractmethod
    def eval(self) -> np.ndarray:
        pass

class CallableExpression(SimpleExpression):
    def __init__(self, output_shape: tuple[int, ...], expr: Callable[[], np.ndarray]):
        super().__init__(output_shape)
        self._expr = expr

    def eval(self) -> np.ndarray:
        value = self._expr()
        if value.shape != self.output_shape:
            raise ShapeMismatchException(
                f"Callable shape: {value.shape}. Expr shape: {self.output_shape}."
            )
        return value
    
class ZeroExpression(SimpleExpression):
    def __init__(self, output_shape: tuple[int, ...]):
        super().__init__(output_shape)

    def eval(self) -> np.ndarray:
        return np.zeros(self.output_shape)