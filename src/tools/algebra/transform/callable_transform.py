import numpy as np
from typing import Callable

from .transform import Transform


class CallableTransform(Transform):
    def __init__(
        self,
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        apply: Callable[[np.ndarray], None],
    ):
        super().__init__(input_shape, output_shape)
        self._callable = apply

    def _apply(self, field: np.ndarray):
        self._callable(field)