import numpy as np
from scipy.sparse.linalg import cg, LinearOperator

from model.discretization import DiscreteBC
from model.domain import Boundary
from algebra.operator import Operator
from algebra.expression import Expression, CallableExpression

class LES():
    def __init__(
        self, lhs: Operator, rhs: Expression
    ):
        self._lhs = lhs 
        self._rhs = rhs 
        self._bcs = dict[Boundary, DiscreteBC]()
        self._rhs_value: np.ndarray | None = None

    def apply_bc(self, bc: DiscreteBC):
        self._bcs[bc.boundary] = bc

    def _assemble(self):
        rhs = self._rhs.eval()
        for bc in self._bcs.values():
            self._lhs = bc.apply_to_les(self._lhs, rhs)
        self._rhs_value = rhs

    def solve(self) -> Expression:
        if self._rhs_value is None:
            self._assemble()
        def cg_solve():
            def matvec(x):
                out = np.zeros_like(x)
                self._lhs.apply(x, out)
                return out
            N = self._lhs.input_shape
            linop = LinearOperator(
                shape=(*N, *N), matvec=matvec
            )
            x, info = cg(linop, self._rhs_value, maxiter=100)
            if info != 0:
                raise RuntimeError(f"CG did not converge, info={info}")
            return x
        return CallableExpression(self._lhs.output_shape, cg_solve)