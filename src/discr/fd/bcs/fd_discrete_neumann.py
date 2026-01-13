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

        grid = self.boundary.grid
        ax = self.boundary.axis
        inward_dir = self.boundary.inward_dir
        h = self.boundary.grid.ax_spacing(ax)

        def modified_apply(field: np.ndarray, out: np.ndarray):
            old_Ax.apply(field, out)
            for id in self.boundary.ids:
                idx = list(grid.idx(id))
                idx[ax] += inward_dir
                inward_id = grid.flat_id(tuple(idx))
                out[id] = (-field[id] + field[inward_id]) / h

        system.rhs[self.boundary.ids] = self.value
        system._Ax = CallableOperator(
            old_Ax.input_shape,
            old_Ax.output_shape,
            modified_apply
        )