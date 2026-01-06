import numpy as np

from tools.algebra import Operator
from tools.algebra.exceptions import ShapeMismatchException
from tools.geometry.grid import StructuredGridND


class FDLaplaceOperator(Operator):
    def __init__(self, grid: StructuredGridND):
        super().__init__(grid.shape, grid.shape)
        self._grid = grid

    def apply(self, input: np.ndarray, out: np.ndarray):
        if input.shape != self.input_shape:
            raise ShapeMismatchException(
                f"Cannot apply to an array. Input array shape: {input.shape}, ",
                "Operator input shape: {self.input_shape}"
            )
        if out.shape != self.output_shape:
            raise ShapeMismatchException(
                f"Cannot apply to an array. Output array shape: {out.shape}, ",
                "Operator input shape: {self.output_shape}"
            )
        ndim = input.ndim
        center = (slice(1, -1),) * ndim
        out = np.zeros_like(out)
        for axis in range(ndim):
            h = self._grid.ax_spacing(axis)
            plus = list(center)
            minus = list(center)
            plus[axis] = slice(2, None)
            minus[axis] = slice(None, -2)
            out[center] += (
                input[tuple(plus)] - 2.0 * input[center] + input[tuple(minus)]
            ) / h**2
