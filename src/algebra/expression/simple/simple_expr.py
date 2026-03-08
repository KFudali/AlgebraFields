from __future__ import annotations
import copy
from abc import ABC, abstractmethod
from typing import Callable
import numpy as np

from algebra.exceptions import ShapeMismatchException
from .operators_mixin import ExpressionOperatorsMixin
from ..expression import Expression

class SimpleExpression(ExpressionOperatorsMixin, Expression):
    def copy(self) -> "SimpleExpression":
        return copy.deepcopy(self)

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

    def copy(self) -> CallableExpression:
        return CallableExpression(self.output_shape, copy.deepcopy(self._expr))

class ZeroExpression(SimpleExpression):
    def __init__(self, output_shape: tuple[int, ...]):
        super().__init__(output_shape)

    def eval(self) -> np.ndarray:
        return np.zeros(self.output_shape)
    
    def copy(self) -> ZeroExpression:
        return ZeroExpression(self.output_shape)