from scipy.sparse.linalg import cg, LinearOperator
from collections.abc import Iterable
from typing import Callable
import numpy as np

import algebra
from .equation import Equation
from discr.core import DiscreteBC

class LES(Equation):
    def __init__(
        self, 
        linop: algebra.operator.CombinedOperator, 
        rhs: algebra.Expression
    ):
        self._linop = linop
        self._rhs = rhs
        self._bcs = set()

    @property
    def bcs(self) -> set[DiscreteBC]:
        return self._bcs

    def add_bcs(self, bcs: Iterable[DiscreteBC]):
        [self._bcs.add(bc) for bc in bcs]

    def solve(self) -> algebra.Expression:
        Ax, rhs = self._assemble()
        def solve() -> np.ndarray:
            x, info = cg(Ax, rhs, maxiter=1000, rtol = 1e-6)
            return x.reshape(self._linop.output_shape)
        return algebra.expression.CallableExpression(self._linop.output_shape, solve)
    
    def _assemble(self) -> tuple[LinearOperator, np.ndarray]:
        rhs = self._rhs.eval()
        linop = self._linop.copy()
        rhs -= linop.take_b().eval()
        for bc in self.bcs:
            bc.apply(linop.Ax.core, rhs)
        matvec = self._assemble_matvec(linop.Ax)
        N = rhs.flatten().shape
        linop = LinearOperator(shape=(*N, *N), matvec=matvec, dtype=float)
        return linop, rhs.flat

    def _assemble_matvec(self, Ax: algebra.Operator) -> Callable:
        def matvec(x: np.ndarray) -> np.ndarray:
            out = np.zeros_like(x)
            Ax.apply(x.reshape(Ax.input_shape), out.reshape(Ax.output_shape))
            return out
        return matvec