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

        ax = self.boundary.axis
        h = self.boundary.grid.ax_spacing(ax)
        inward_ids = []
        for id in self.boundary.ids:
            idx = list(self.boundary.grid.idx(id))
            idx[ax] += self.boundary.inward_dir
            inward_id = self.boundary.grid.flat_id(tuple(idx))
            inward_ids.append(inward_id)

        ids = self.boundary.ids
        def modified_apply(field: np.ndarray, out: np.ndarray):
            old_Ax.apply(field, out)
            out[ids] = (-field[ids] + field[inward_ids]) / h**2

        system._Ax = CallableOperator(
            old_Ax.input_shape,
            old_Ax.output_shape,
            modified_apply
        )
        system.rhs[self.boundary.ids] += self.value / h