import numpy as np

from .boundary_condition import BoundaryCondition, Space
from model.domain.boundary import Boundary
from spacer.base import Space

class DirichletBC(BoundaryCondition):
    def __init__(
        self,
        space: Space,
        boundary: Boundary,
        value: float
    ):
        super().__init__(space, boundary)
        self.value = value

    def apply_linear(self, A: np.ndarray, b: np.ndarray):
        self.space.disc.bcs.dirichlet.apply_linear(
            A, b, self.value, self._boundary
        )