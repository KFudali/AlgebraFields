from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Self
import numpy as np

from algebra.exceptions import ShapeMismatchException
from algebra.expression import Expression


class Operator(ABC):
    def __init__(
        self,
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
    ):
        self._input_shape = input_shape
        self._output_shape = output_shape

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self._input_shape

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self._output_shape

    def apply(self, field: np.ndarray, out: np.ndarray):
        if field.shape != self.input_shape:
            raise ShapeMismatchException(
                "Cannot apply operator. Input size differs from operator size. "
                f"Field shape: {field.shape}, "
                f"Operator input shape: {self.input_shape}"
            )
        if out.shape != self.output_shape:
            raise ShapeMismatchException(
                "Cannot apply operator. Output size differs from operator size. "
                f"Out shape: {out.shape}, "
                f"Operator output shape: {self.output_shape}"
            )
        self._apply(field, out)

    @abstractmethod
    def _apply(self, field: np.ndarray, out: np.ndarray):
        pass

    @abstractmethod
    def __neg__(self) -> Self:
        pass

    @abstractmethod
    def __add__(self, other: Operator) -> Self:
        pass

    @abstractmethod
    def __mul__(self, other: float) -> Self:
        pass

    @abstractmethod
    def __truediv__(self, other: float) -> Self:
        pass

    def __sub__(self, other: Operator) -> Self:
        return self + (-other)

    def __rmul__(self, other: float | Expression) -> Self:
        return self * other
