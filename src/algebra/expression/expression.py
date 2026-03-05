from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Self
import numpy as np

from .scalar_expression import ScalarExpression

class Expression(ABC):
    def __init__(self, output_shape: tuple[int, ...]):
        self._output_shape = output_shape

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self._output_shape

    @abstractmethod
    def eval(self) -> np.ndarray:
        pass

    @abstractmethod
    def __neg__(self) -> Self:
        pass

    @abstractmethod
    def __add__(self, other: Expression | ScalarExpression | float ) -> Self:
        pass

    @abstractmethod
    def __mul__(self, other: Expression | ScalarExpression | float) -> Self:
        pass

    @abstractmethod
    def __truediv__(self, other: Expression | ScalarExpression | float) -> Self:
        pass

    def __radd__(self, other: Expression | ScalarExpression | float) -> Self:
        return self + other

    def __sub__(self, other: Expression | ScalarExpression | float) -> Self:
        return self + (-other)

    def __rsub__(self, other: Expression | ScalarExpression | float) -> Self:
        return (-self) + other

    def __rmul__(self, other: Expression | ScalarExpression | float) -> Self:
        return self * other