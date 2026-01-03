from model.domain.boundary import Boundary

from ..shapebound import ShapeBound, FieldShape
from .dirichlet_condition import DirichletBC
from .neumann_condition import NeumannBC

class BCFactory(ShapeBound):
    def __init__(
        self, shape: FieldShape
    ):
        super().__init__(shape)
    
    def dirichlet(self, boundary: Boundary, value: float) -> DirichletBC:
        return DirichletBC(self.shape, boundary, value)
    
    def neumann(self, boundary: Boundary, value: float) -> NeumannBC:
        return NeumannBC(self.shape, boundary, value)