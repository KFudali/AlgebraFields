from abc import ABC, abstractmethod
import numpy as np


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

    @abstractmethod
    def apply(self, field: np.ndarray, out: np.ndarray):
        pass