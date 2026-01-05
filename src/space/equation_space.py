from .core import Space
from .field import Field
from model.discrete import Discretization, DiscreteBCs, DiscreteOperators



class EquationSpace(Space):
    def __init__(self, discretization: Discretization):
        super().__init__(discretization)

    def field(self, components: int = 1) -> Field:
        return Field(space=self, components=components)

    @property
    def operators(self) -> DiscreteOperators:
        return self._discretization.operators

    @property
    def bcs(self) -> DiscreteBCs:
        return self._discretization.bcs