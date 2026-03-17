from scipy.sparse.linalg import cg, LinearOperator
from collections.abc import Iterable
from typing import Callable
import numpy as np

import algebra
from algebra.operator import CombinedOperator
from .equation import Equation
from discr.core import DiscreteBC

class LES(Equation):
    def __init__(
        self, 
        linop: algebra.Operator,
        rhs: algebra.Expression
    ):
        self._linop = linop
        self._rhs = rhs
        self._bcs = list[DiscreteBC]()


    @property
    def bcs(self) -> list[DiscreteBC]:
        return self._bcs

    def add_bcs(self, bcs: Iterable[DiscreteBC]):
        [self._bcs.append(bc) for bc in bcs if bc not in self._bcs]

    def solve(self) -> algebra.Expression:
        Ax, rhs = self._assemble()
        def solve() -> np.ndarray:
            x, info = cg(Ax, rhs, maxiter=1000, rtol = 1e-6)
            x = x.reshape(self._linop.output_shape)
            [bc.post_solve(x[0]) for bc in self.bcs]
            return x
        return algebra.expression.CallableExpression(self._linop.output_shape, solve)
    
    def _assemble(self) -> tuple[LinearOperator, np.ndarray]:
        rhs = self._rhs.eval()
        linop = self._linop.copy()
        Ax = linop.core
        if isinstance(Ax, CombinedOperator):
            rhs -= Ax.take_b().eval()
            Ax = Ax.Ax.core
        for bc in self.bcs:
            bc.apply(Ax, rhs[0])
        matvec = self._assemble_matvec(linop)
        N = rhs.flatten().shape
        linop = LinearOperator(shape=(*N, *N), matvec=matvec, dtype=float)
        return linop, rhs.flat

    def _assemble_matvec(self, Ax: algebra.Operator) -> Callable:
        def matvec(x: np.ndarray) -> np.ndarray:
            out = np.zeros_like(x)
            Ax.apply(x.reshape(Ax.input_shape), out.reshape(Ax.output_shape))
            return out
        return matvec