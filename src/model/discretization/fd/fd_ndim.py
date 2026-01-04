from model.discretization.core import Discretization
from .domain import FDDomain
from .fd_operators import FDOperators
from .bcs import FDDiscreteBCs

class FDDiscretization(Discretization[FDDomain]):
    def __init__(self, domain: FDDomain):
        super().__init__()
        self._domain = domain
        self._operators = FDOperators(domain)
        self._bcs = FDDiscreteBCs()
   
    @property
    def shape(self) -> tuple[int, ...]:
        return self._domain.grid.shape
    
    @property
    def domain(self) -> FDDomain:
        return self._domain
    
    @property
    def operators(self) -> FDOperators:
        return self._operators

    @property
    def bcs(self) -> FDDiscreteBCs:
        return self._bcs