import numpy as np

from model.discrete.core.bcs import DiscreteDirichletBC
from ..domain import FDBoundary
from algebra.operator import Operator, CallableOperator
from algebra.system import LES

class FDDirichletBC(DiscreteDirichletBC):
    def __init__(self, boundary: FDBoundary, value: float):
        super().__init__(boundary, value)
    
    @property
    def boundary(self) -> FDBoundary:
        return self._boundary

    def apply_to_system(self, system: LES):
        def modified_apply(field: np.ndarray, out: np.ndarray):
            system.Ax.apply(field, out)
            out[self.boundary.ids] = self._value
        Ax = CallableOperator(
            system.Ax.input_shape, system.Ax.output_shape, modified_apply
        )
        system._Ax = Ax
        system._rhs[self.boundary.ids] = self._value