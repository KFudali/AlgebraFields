from abc import ABC, abstractmethod

from discr.core import DiscreteBC
from discr.fd.domain import FDBoundary
from discr.fd.operators import FDStencilOperator


class FDDiscreteBC(DiscreteBC[FDBoundary], ABC):
    def __init__(self, boundary: FDBoundary):
        super().__init__(boundary)

    @abstractmethod
    def apply(self, operator: FDStencilOperator): pass