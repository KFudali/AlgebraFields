from ..discretization import Discretization
import numpy as np
from .domain import FDDomain
from .fd_operators import FDOperators
from .fd_stencils import FDStencils
from .bc import FDDiscreteBCs

class FDDiscretization(Discretization[FDDomain]):
    def __init__(self, domain: FDDomain):
        super().__init__()
        self._domain = domain
        self._stencils = FDStencils(domain)
        self._operators = FDOperators(domain)
        self._bcs = FDDiscreteBCs()
   
    @property
    def shape(self) -> tuple[int, ...]:
        return self._domain.grid.shape
    
    def zeros(self) -> np.ndarray:
        return np.zeros(self._domain.grid.shape)

    @property
    def domain(self) -> FDDomain:
        return self._domain

    @property
    def stencils(self) -> FDStencils:
        return self._stencils
    
    @property
    def operators(self) -> FDOperators:
        return self._operators

    @property
    def bcs(self) -> FDDiscreteBCs:
        return self._bcs

    @property
    def dim(self) -> int:
        self._domain.grid.shape[0]