import numpy as np
import copy

from .fd_discrete_bc import FDDiscreteBC, FDBoundary, FDStencilOperator
import algebra


class FDDiscreteNeumann(FDDiscreteBC):
    def __init__(self, boundary: FDBoundary, value: float):
        super().__init__(boundary)
        self._value = value

    @property
    def value(self) -> float:
        return self._value

    def apply(self, op: FDStencilOperator, rhs: np.ndarray):
        stencil = op.boundary_stencils[self.boundary.id]
        contribs = stencil.contribs[self.boundary.axis]
        h = op.domain.grid.ax_spacing(self.boundary.axis)
        for offset, value in contribs.copy().items():
            if offset * self.boundary.inward_dir < 0:
                contrib = value
                contribs[-offset] = contribs.get(-offset, 0.0) + contrib
                contribs.pop(offset)
                rhs.flat[self.boundary.region] = -contrib * 2 * h * self.value