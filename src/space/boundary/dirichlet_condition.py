from .boundary_condition import BoundaryCondition, Space
from domain import Boundary

class DirichletBC(BoundaryCondition):
    def __init__(
        self,
        space: Space,
        boundary: Boundary,
        value: float
    ):
        super().__init__(space, boundary)
        self.value = value