from model.discrete.core.discrete_bcs import DiscreteBCs
from .fd_discrete_dirichlet import FDDirichletBC
from .fd_discrete_neumann import FDNeumannBC
from model.domain import Boundary
from model.discrete.core.bcs import DiscreteDirichletBC, DiscreteNeumannBC

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
