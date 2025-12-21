from .boundary_condition import BoundaryCondition, Space
from model.domain.boundary import Boundary
from space.base import Space

class DirichletBC(BoundaryCondition):
    def __init__(
        self,
        space: Space,
        boundary: Boundary,
        value: float
    ):
        super().__init__(space, boundary)
        self.value = value