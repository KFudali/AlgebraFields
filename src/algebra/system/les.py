import numpy as np
from ..operator import Operator
from ..exceptions import ShapeMismatchException


class LES():
    def __init__(
        self, Ax: Operator, rhs: np.ndarray
    ):
        self._Ax = Ax
        self._rhs = rhs
        if self._Ax.output_shape != self._rhs.shape:
            raise ShapeMismatchException()
    
    @property
    def Ax(self) -> Operator: return self._Ax

    @property
    def rhs(self) -> np.ndarray: return self._rhs

    def solve(self):
    def matvec(x: np.ndarray):
            out = np.zeros_like(x)
            Ax.apply(x, out)
            return out

        N = rhs.size
        linop = LinearOperator(shape=(N, N), matvec=matvec, dtype=rhs.dtype)

        solver_map = {
            'cg': cg,
            'bicg': bicg,
            'bicgstab': bicgstab,
            'gmres': gmres,
            'minres': minres
        }

        if self._method not in solver_map:
            raise ValueError(f"Unknown solver: {self._method}")

        solver = solver_map[self._method]
        x, info = solver(linop, rhs, maxiter=self._maxiter, tol=self._tol)

        if info != 0:
            raise MaxIterationReached(
                f"{self._method} did not converge, info={info}"
            )
        return x