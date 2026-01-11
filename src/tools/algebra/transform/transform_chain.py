import numpy as np

from .transform import Transform
from algebra.exceptions import ShapeMismatchException

class TransformChain(Transform):
    def __init__(self, first_tr: Transform):
        super().__init__(first_tr.input_shape, first_tr.output_shape)
        self._trs = list[Transform]()

    def append(self, tr: Transform):
        if self._trs[-1].output_shape != tr.input_shape:
            raise ShapeMismatchException(
                ("Appended transform input_shape does not match previous transform ",
                f"output_shape.",
                f"Appended: {tr.input_shape}, Prev: {self._trs[-1].output_shape}")
            )
        self._trs.append(tr)
        self._output_shape = tr.output_shape

    def _apply(self, field: np.ndarray):
        for tr in self._trs:
            tr.apply(field)