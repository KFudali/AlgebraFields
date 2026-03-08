from discr.core import DiscreteBCFactory
from discr.core.domain import BoundaryId
from ..domain import FDDomain

from .fd_neumann import FDDiscreteNeumann
from .fd_dirichlet import FDDiscreteDirichlet

class FDDiscreteBCFactory(DiscreteBCFactory):
    def __init__(self, domain: FDDomain):
        super().__init__()
        self._domain = domain

    def dirichlet(self, bid: BoundaryId, value: float) -> FDDiscreteDirichlet:
        boundary = self._domain.boundary(bid)
        return FDDiscreteDirichlet(boundary, value)

    def neumann(self, bid: BoundaryId, value: float) -> FDDiscreteNeumann:
        boundary = self._domain.boundary(bid)
        return FDDiscreteNeumann(boundary, value)