import numpy as np
import copy

from .fd_discrete_bc import FDDiscreteBC, FDBoundary, FDStencilOperator
import algebra

class FDDiscreteDirichlet(FDDiscreteBC):
    def __init__(self, boundary: FDBoundary, value: float):
        super().__init__(boundary)
        self._value = value

    @property
    def value(self) -> float:
        return self._value

    def apply(
        self, op: FDStencilOperator
    ) -> tuple[FDStencilOperator, algebra.expression.CallableExpression]:
        op = copy.deepcopy(op)
        stencil = op.boundary_stencils[self.boundary.id]

        for ax in stencil.contribs.keys():
            if ax != self.boundary.axis:
                stencil.contribs[ax] = {0: 0}
        
        field = np.zeros(op.input_shape)
        field[self.boundary.region] = self.value
        dirichlet_contrib = np.zeros(op.output_shape)
        stencil.apply_to_region(field, dirichlet_contrib, self.boundary.region)

        stencil.contribs[self.boundary.axis] = {0: 1}
        def return_lifting() -> np.ndarray:
            return dirichlet_contrib
        contrib = algebra.expression.CallableExpression(op.output_shape, return_lifting)
        return op, contrib