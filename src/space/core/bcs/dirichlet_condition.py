import numpy as np

from ..shapebound import ShapeBound, FieldShape
from .boundary_condition import BoundaryCondition
from model.domain.boundary import Boundary

class DirichletBC(BoundaryCondition):
    def __init__(
        self, shape: FieldShape, boundary: Boundary, value: float
    ):
        super().__init__(shape, boundary)
        self._value = value

    @property
    def value(self) -> float:
        return self._value

    def apply_linear(self, A: np.ndarray, b: np.ndarray):
        self.space.discretization.bcs.dirichlet.apply_linear()


    bc.apply_linear(boundary, value, in, out)

    (in, out)