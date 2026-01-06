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
        axis = self.boundary.axis
        inward_dir = self.boundary.inward_dir
        h = self.boundary.grid.ax_spacing(axis)
        shape = self.boundary.grid.shape

        def modified_apply(field: np.ndarray, out: np.ndarray):
            system.Ax.apply(field, out)

            for i in self.boundary.ids:
                idx = list(np.unravel_index(i, shape))
                idx_in = idx.copy()
                idx_in[axis] += inward_dir
                j = np.ravel_multi_index(idx_in, shape)

                out[i] = (field[j] - field[i]) / h - self._value
            Ax = CallableOperator(
                system.Ax.input_shape, system.Ax.output_shape, modified_apply
            )
            system._Ax = Ax
