from discr.core.bcs import DiscreteBC
from ..domain import FDBoundary

class FDDiscreteBC(DiscreteBC):
    def __init__(self, boundary: FDBoundary, value: float):
        super().__init__(boundary)
        self._value = value
        self._boundary = boundary