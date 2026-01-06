from .core import Space
from .field import Field
from discr import Discretization, DiscreteOperators, bcs



class EquationSpace(Space):
    def __init__(self, discretization: Discretization):
        super().__init__(discretization)

    def field(self, components: int = 1) -> Field:
        return Field(space=self, components=components)

    @property
    def operators(self) -> DiscreteOperators:
        return self._discretization.operators

    @property
    def bcs(self) -> bcs.DiscreteBCs:
        return self._discretization.bcs