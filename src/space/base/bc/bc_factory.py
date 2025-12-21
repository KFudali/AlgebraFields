from space.base import SpaceBound, Space
from model.domain.boundary import Boundary

from .dirichlet_condition import DirichletBC
from .neumann_condition import NeumannBC

class BCFactory(SpaceBound):
    def __init__(
        self, 
        space: Space
    ):
        super().__init__(space)
    
    def dirichlet(self, boundary: Boundary, value: float) -> DirichletBC:
        return DirichletBC(self.space, boundary, value)
    
    def neumann(self, boundary: Boundary, value: float) -> NeumannBC:
        return NeumannBC(self.space, boundary, value)