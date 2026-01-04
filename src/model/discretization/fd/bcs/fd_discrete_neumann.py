
import numpy as np

from model.discretization.core import DiscreteNeumannBC
from algebra.expression import Expression
from algebra.operator import Operator, CallableOperator

from ..domain import FDBoundary


class FDNeumannBC(DiscreteNeumannBC):
    def __init__(self, boundary: FDBoundary, value: float):
        super().__init__()
        self._value = value
        self._boundary = boundary

    def apply_to_les(self, op: Operator, rhs: np.ndarray) -> Operator:
        axis = self._boundary.axis
        inward_dir = self._boundary.inward_dir
        h = self._boundary.grid.ax_spacing(axis)
        shape = self._boundary.grid.shape
        def modified_apply(A: np.ndarray, b: np.ndarray):
            for i in self._boundary.ids:
                A[i, :] = 0.0

                idx = list(np.unravel_index(i, shape))
                idx_in = idx.copy()
                idx_in[axis] += inward_dir
                j = np.ravel_multi_index(idx_in, shape)

                A[i, i] = -inward_dir / h
                A[i, j] =  inward_dir / h
                b[i] = self._value
        return CallableOperator(op.input_shape, op.output_shape, modified_apply)
