from abc import ABC, abstractmethod

from discretization.core import DiscreteBC
from discretization.fd.domain import FDBoundary
from discretization.fd.operators import FDStencilOperator


class FDDiscreteBC(DiscreteBC[FDBoundary], ABC):
    def __init__(self, boundary: FDBoundary):
        super().__init__(boundary)

    @abstractmethod
    def apply(self, operator: FDStencilOperator): pass