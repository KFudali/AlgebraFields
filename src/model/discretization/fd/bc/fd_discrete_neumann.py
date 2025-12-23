
import numpy as np

from ...boundaries import DiscreteNeumannBC
from ..domain import FDBoundary

class FDNeumannBC(DiscreteNeumannBC):
    def __init__(self):
        super().__init__()

    def apply_linear(
        self, A: np.ndarray, b: np.ndarray, value: float, boundary: FDBoundary
    ):
        axis = boundary.axis
        inward_dir = boundary.inward_dir
        h = boundary.grid.ax_spacing(axis)
        shape = boundary.grid.shape

        for i in boundary.ids:
            A[i, :] = 0.0

            idx = list(np.unravel_index(i, shape))
            idx_in = idx.copy()
            idx_in[axis] += inward_dir
            j = np.ravel_multi_index(idx_in, shape)

            A[i, i] = -inward_dir / h
            A[i, j] =  inward_dir / h
            b[i] = value