import numpy as np
from ..operator import Operator
from ..exceptions import ShapeMismatchException, SolverMaxIterReached
from scipy.sparse.linalg import cg, LinearOperator


class LES:
    def __init__(self, Ax: Operator, rhs: np.ndarray):
        self._Ax = Ax
        self._rhs = rhs
        if self._Ax.input_shape != self._rhs.shape:
            raise ShapeMismatchException(
                f"Input shape mismatch: Ax expects input shape {self._Ax.input_shape}, "
                f"but rhs has shape {self._rhs.shape}."
            )

        if self._Ax.output_shape != self._rhs.shape:
            raise ShapeMismatchException(
                f"Output shape mismatch: Ax produces output shape {self._Ax.output_shape}, "
                f"but rhs has shape {self._rhs.shape}."
            )

    @property
    def Ax(self) -> Operator:
        return self._Ax

    @property
    def rhs(self) -> np.ndarray:
        return self._rhs

    def solve(
        self, method: str = "cg", maxiter: int = 100, tol: float = 1e-8
    ) -> np.ndarray:
        Ax = self._Ax
        rhs = self._rhs

        def matvec(x: np.ndarray):
            out = np.zeros_like(x)
            Ax.apply(x, out)
            return out

        N = rhs.size
        linop = LinearOperator(shape=(N, N), matvec=matvec, dtype=rhs.dtype)

        solver_map = {
            "cg": cg,
            # 'bicg': bicg,
            # 'bicgstab': bicgstab,
            # 'gmres': gmres,
            # 'minres': minres
        }

        if method not in solver_map:
            raise ValueError(f"Unknown solver: {method}")

        solver = solver_map[method]
        x, info = solver(linop, rhs, maxiter=maxiter, tol=tol)

        if info != 0:
            raise SolverMaxIterReached(f"{method} did not converge, info={info}")
        return x
