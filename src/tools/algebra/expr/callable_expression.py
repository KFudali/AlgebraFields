from typing import Callable
import numpy as np

from tools.algebra.exceptions import ShapeMismatchException
from .expression import Expression


class CallableExpression(Expression):
    def __init__(self, output_shape: tuple[int, ...], expr: Callable[[], np.ndarray]):
        self._output_shape = output_shape
        self._expr = expr

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self._output_shape

    def eval(self) -> np.ndarray:
        value = self._expr()
        if value.shape != self.output_shape:
            raise ShapeMismatchException(
                f"Callable shape: {value.shape}. Expr shape: {self.output_shape}."
            )