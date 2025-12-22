import numpy as np
from ..discretization import DiscreteOperators
from .domain import FDDomain
from model.geometry.grid import StructuredGridND

class FDOperators(DiscreteOperators):
    def __init__(self, domain: FDDomain):
        self._domain = domain
        super().__init__()

    @property
    def grid(self) -> StructuredGridND:
        return self._domain.grid

    def laplace(self, values: np.ndarray, laplace: np.ndarray):
        assert values.shape == laplace.shape
        laplace.fill(0.0)
        if self.grid.ndim == 2:
            dx = self.grid.ax_spacing(0)
            dy = self.grid.ax_spacing(1)
            laplace[1:-1, 1:-1] = (
                (values[2:, 1:-1] - 2*values[1:-1, 1:-1] + values[:-2, 1:-1]) / dx**2
              + (values[1:-1, 2:] - 2*values[1:-1, 1:-1] + values[1:-1, :-2]) / dy**2
            )
            return
        else:
            ndim = values.ndim
            center = (slice(1, -1),) * ndim
            for axis in range(ndim):
                h = self.grid.ax_spacing(axis)
                plus  = list(center)
                minus = list(center)
                plus[axis]  = slice(2, None)
                minus[axis] = slice(None, -2)

                laplace[center] += (
                    values[tuple(plus)]
                    - 2.0 * values[center]
                    + values[tuple(minus)]
                ) / h**2