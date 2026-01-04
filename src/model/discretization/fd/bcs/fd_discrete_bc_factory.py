from model.core.discrete_bcs import DiscreteBCs
from .fd_discrete_dirichlet import FDDirichletBC
from .fd_discrete_neumann import FDNeumannBC

class FDDiscreteBCs(DiscreteBCs):
    def __init__(self):
        super().__init__()
        self._dirichlet = FDDirichletBC()
        self._neumann = FDNeumannBC()

    @property
    def dirichlet(self) -> FDDirichletBC:
        return self._dirichlet
    
    @property
    def neumann(self) -> FDNeumannBC:
        return self._neumann