import numpy as np

from discr.core.bcs import DiscreteDirichletBC
from discr.fd.domain import FDBoundary

from tools.algebra.operator import CallableOperator
from tools.algebra.system import LES


class FDDirichletBC(DiscreteDirichletBC):
    def __init__(self, boundary: FDBoundary, value: float):
        super().__init__(boundary, value)
    
    @property
    def boundary(self) -> FDBoundary:
        return self._boundary

    def apply_to_system(self, system: LES):
        old_Ax = system.Ax
        ids = self.boundary.ids
        value = self._value
        

        def modified_apply(x: np.ndarray, out: np.ndarray):
            old_Ax.apply(x, out)
            out[ids] = x[ids]

        rhs = np.zeros_like(system.rhs)
        rhs[ids] = value
        mod_rhs = np.zeros_like(system.rhs)
        system.Ax.apply(rhs, mod_rhs)
        system._rhs[:] -= mod_rhs[:]
        system._rhs[ids] = 0
        system._Ax = CallableOperator(
            old_Ax.input_shape,
            old_Ax.output_shape,
            modified_apply
        )