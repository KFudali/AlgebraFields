import numpy as np
from scipy.sparse.linalg import LinearOperator, cg, bicg, bicgstab, gmres, minres
from .expression import Expression
from ..system import LES

class MaxIterationReached(Exception): pass

class SolveLESExpression(Expression):
    def __init__(self, les: LES, method='cg', maxiter=100, tol=1e-8):
        super().__init__(les.Ax.output_shape)
        self._system = les
        self._method = method
        self._maxiter = maxiter
        self._tol = tol

    @property
    def system(self) -> LES:
        return self._system

    def eval(self) -> np.ndarray:
        Ax = self._system.Ax
        rhs = self._system.rhs
    