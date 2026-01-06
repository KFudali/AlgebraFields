from .discrete_bc import DiscreteBC
from discr.core.domain import BoundaryType

class DiscreteDirichletBC(DiscreteBC[BoundaryType]):
    def __init__(self, boundary: BoundaryType, value: float):
        super().__init__(boundary)
        self._value = value

    @property
    def value(self) -> float:
        return self._value