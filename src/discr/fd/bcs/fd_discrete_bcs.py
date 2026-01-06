from discr.core.bcs import DiscreteBCs, DiscreteDirichletBC, DiscreteNeumannBC
from discr.core.domain import Boundary

from .fd_discrete_dirichlet import FDDirichletBC
from .fd_discrete_neumann import FDNeumannBC

class FDDiscreteBCs(DiscreteBCs):
    def __init__(self):
        super().__init__()

    def dirichlet(
        self, boundary: Boundary, value: float
    ) -> DiscreteDirichletBC:
        return FDDirichletBC(boundary, value)
    
    def neumann(
        self, boundary: Boundary, value: float
    ) -> DiscreteNeumannBC:
        return FDNeumannBC(boundary, value)
