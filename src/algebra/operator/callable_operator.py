from typing import Callable
import numpy as np

from .operator import Operator


class CallableOperator(Operator):
    def __init__(
        self,
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        apply: Callable[[np.ndarray, np.ndarray],],
    ):
        super().__init__(input_shape, output_shape)
        self._apply_callable = apply

    def apply(self, field: np.ndarray, out: np.ndarray):
        self._apply_callable(field, out)