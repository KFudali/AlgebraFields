import numpy as np
import copy
from tools import region
from .fd_discrete_bc import FDDiscreteBC, FDBoundary, FDStencilOperator
import algebra

class FDDiscreteDirichlet(FDDiscreteBC):
    def __init__(self, boundary: FDBoundary, value: float):
        super().__init__(boundary)
        self._value = value

    @property
    def value(self) -> float:
        return self._value

    def apply(self, op: FDStencilOperator, rhs: np.ndarray):
        op.resolve_factor()
        stencil = op.boundary_stencils[self.boundary.id]
        for ax in stencil.contribs.keys():
            if ax != self.boundary.axis:
                stencil.contribs[ax] = {0: 0}
        
        field = np.zeros(op.input_shape)
        field[self.boundary.region] = self.value
        dirichlet_contrib = np.zeros(op.output_shape)
        ax_range = stencil.ax_range(self.boundary.axis, self.boundary.inward_dir)
        offsets = [0 for ax in range(self.boundary.grid.ndim)]
        ranges = [ax_range for i in range(self.boundary.grid.ndim)]
        offsets[self.boundary.axis] = tuple(ranges)
        boundary_interior = region.interior(self.boundary.grid.shape, tuple(offsets))

        stencil.apply_to_region_on_ax(
            field, 
            dirichlet_contrib, 
            boundary_interior,
            self.boundary.axis 
        )
        stencil.contribs[self.boundary.axis] = {0: 1}
        rhs -= dirichlet_contrib