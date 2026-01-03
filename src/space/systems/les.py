import numpy as np
from scipy.sparse.linalg import cg, LinearOperator

from space.core import Operator, Expression, bcs
from space.expr import GetterExpression

class LES():
    def __init__(
        self, lhs: Operator, rhs: Expression
    ):
        self._lhs = lhs 
        self._rhs = rhs 
        self._bcs = set[bcs.BoundaryCondition]()
        self._rhs_value: np.ndarray | None = None

    def apply_bc(self, bc: bcs.BoundaryCondition):
        self._bcs.add(bc)

    def _assemble(self):
        rhs = self._rhs._eval()
        for bc in self._bcs:
            self._lhs = bc.apply_linear(self._lhs, rhs)
        self._rhs_value = rhs

    def solve(self) -> Expression:
        if self._rhs_value is None:
            self._assemble()
        def cg_solve():
            def matvec(x):
                out = np.zeros_like(x)
                self._lhs._apply(x, out)
                return out
            linop = LinearOperator(
                shape=self._lhs.array_shape, matvec=matvec
            )
            x, info = cg(linop, self._rhs_value, maxiter=100)
            if info != 0:
                raise RuntimeError(f"CG did not converge, info={info}")
            return x
        return GetterExpression(self._lhs.shape, cg_solve)