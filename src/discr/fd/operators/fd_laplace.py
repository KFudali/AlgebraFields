import numpy as np

from tools.algebra import Operator
from tools.algebra.exceptions import ShapeMismatchException
from tools.geometry.grid import StructuredGridND


class FDLaplaceOperator(Operator):
    def __init__(self, grid: StructuredGridND):
        super().__init__(grid.shape, grid.shape)
        self._grid = grid

    def _apply(self, field: np.ndarray, out: np.ndarray):
        if field.shape != self.input_shape:
            raise ShapeMismatchException(
                f"Cannot apply to an array. Input array shape: {field.shape}, ",
                f"Operator input shape: {self.input_shape}"
            )
        if out.shape != self.output_shape:
            raise ShapeMismatchException(
                f"Cannot apply to an array. Output array shape: {out.shape}, ",
                f"Operator input shape: {self.output_shape}"
            )
        ndim = field.ndim
        center = (slice(1, -1),) * ndim
        out[:] = 0 
        for axis in range(ndim):
            h = self._grid.ax_spacing(axis)
            plus = list(center)
            minus = list(center)
            plus[axis] = slice(2, None)
            minus[axis] = slice(None, -2)
            out[center] += (
                field[tuple(plus)] - 2.0 * field[center] + field[tuple(minus)]
            ) / h**2
        out[np.unravel_index(self._grid.boundary_ids, self._grid.shape)] = 0