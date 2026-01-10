import numpy as np

from discr.core.bcs import DiscreteNeumannBC
from discr.fd.domain import FDBoundary

from tools.algebra.operator import CallableOperator
from tools.algebra.system import LES


class FDNeumannBC(DiscreteNeumannBC):
    def __init__(self, boundary: FDBoundary, value: float):
        super().__init__(boundary, value)

    @property
    def boundary(self) -> FDBoundary:
        return self._boundary

    def apply_to_system(self, system: LES):
        old_Ax = system.Ax

        axis = self.boundary.axis
        inward_dir = self.boundary.inward_dir
        h = self.boundary.grid.ax_spacing(axis)
        shape = self.boundary.grid.shape
        ids = self.boundary.ids
        value = self._value

        def modified_apply(field: np.ndarray, out: np.ndarray):
            old_Ax.apply(field, out)

            for i in ids:
                idx = list(np.unravel_index(i, shape))
                idx_in = idx.copy()
                idx_in[axis] += inward_dir
                j = np.ravel_multi_index(idx_in, shape)

                out[i] = (field[j] - field[i]) / h - value

        system._Ax = CallableOperator(
            old_Ax.input_shape,
            old_Ax.output_shape,
            modified_apply
        )