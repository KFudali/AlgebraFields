from abc import ABC, abstractmethod
import numpy as np
from ..exceptions import ShapeMismatchException

class Operator(ABC):
    def __init__(self, input_shape: tuple[int, ...], output_shape: tuple[int, ...]):
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
                f"Output shape: {field.shape}, ",
                f"Operator input shape: {self.input_shape}"
            )
        if out.shape != self.output_shape:
            raise ShapeMismatchException(
                "Cannot apply operator. Output size differs from operator size. "
                f"Output shape: {out.shape}, ",
                f"Operator output shape: {self.output_shape}"
            )
        self._apply(field, out)

    @abstractmethod
    def _apply(self, field: np.ndarray, out: np.ndarray): pass