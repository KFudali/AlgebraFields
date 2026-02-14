from discr.core import DiscreteOperators
from tools.geometry.grid import StructuredGridND

from .domain import FDDomain
from .operators import FDLaplaceOperator

class FDOperators(DiscreteOperators):
    def __init__(self, domain: FDDomain):
        self._domain = domain
        super().__init__()

    @property
    def grid(self) -> StructuredGridND:
        return self._domain.grid

    def laplace(self) -> FDLaplaceOperator:
        return FDLaplaceOperator(self.grid)