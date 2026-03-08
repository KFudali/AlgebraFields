from discr.core import DiscreteBCFactory
from .fd_discrete_bc import FDBoundary

from .fd_neumann import FDDiscreteNeumann
from .fd_dirichlet import FDDiscreteDirichlet

class FDDiscreteBCFactory(DiscreteBCFactory[FDBoundary]):
    def __init__(self):
        super().__init__()

    def dirichlet(self, boundary: FDBoundary, value: float) -> FDDiscreteDirichlet:
        return FDDiscreteDirichlet(boundary, value)

    def neumann(self, boundary: FDBoundary, value: float) -> FDDiscreteNeumann:
        return FDDiscreteNeumann(boundary, value)