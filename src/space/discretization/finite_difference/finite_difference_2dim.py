from ..discretization import Discretization
from ....mesh.structured import StructuredGrid2D
import numpy as np
from typing import Callable

#TODO: Generalize into NDims - its doable for finite difference
class FDDiscretization2D(Discretization):
    def __init__(
        self,
        structured_grid: StructuredGrid2D
    ):
        super().__init__()
        self._grid = structured_grid

    @property
    def grid(self) -> StructuredGrid2D:
        return self._grid

    @property
    def grad(self) -> np.ndarray:
        nx, ny = self.grid.nx, self.grid.ny
        N = nx * ny

        left  = np.eye(N, N, -1)
        right = np.eye(N, N, 1)
        up    = np.eye(N, N, -nx)
        down  = np.eye(N, N, nx)

        grad_x = right - left

        gx = grad_x.reshape(-1)
        gx[self.grid.left_ids] = -1
        gx[self.grid.right_ids] = 1
        grad_x = gx.reshape(N, N) / self.grid.dx

        grad_y = down - up
        gy = grad_y.reshape(-1)
        gy[self.grid.top_ids] = -1
        gy[self.grid.bottom_ids] = 1
        grad_y = gy.reshape(N, N) / self.grid.dy

        return np.stack([grad_x, grad_y])

    @property
    def laplace(self):
        return self.div @ self.grad

    @property
    def div(self) -> np.ndarray:
        G = self.grad
        return -np.concatenate([g.T for g in G], axis=1)

    @property
    def curl(self) -> np.ndarray:
        Dx, Dy = self.grad
        return np.concatenate([-Dy, Dx], axis=1)

    @property
    def mass(self) -> np.ndarray:
        N = self.grid.nx * self.grid.ny
        return np.eye(N, N) * (self.grid.dx * self.grid.dy)

    def integrate(
        self, 
        f: Callable[[np.ndarray], np.ndarray], 
        domain_id: int = None
    ) -> float:
        nx, ny = self.grid.nx, self.grid.ny
        hx = 1.0 / (nx-1)
        hy = 1.0 / (ny-1)
        N = nx * ny

        weights = np.full((nx, ny), hx * hy)

        weights.reshape(-1)
        weights[self.grid.left_ids]   *= 0.5
        weights[self.grid.right_ids]  *= 0.5
        weights[self.grid.top_ids]   *= 0.5
        weights[self.grid.bottom_ids]  *= 0.5
        weights.reshape(nx, ny)

        weights[0,0]     *= 0.5
        weights[0,-1]    *= 0.5
        weights[-1,0]    *= 0.5
        weights[-1,-1]   *= 0.5

        w = weights.ravel()

        return np.sum(f * w)


