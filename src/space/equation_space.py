from .core import Space, FieldShape
from .field import Field
from model.discretization import Discretization, DiscreteBCs, DiscreteOperators



class EquationSpace(Space):
    def __init__(self, discretization: Discretization):
        super().__init__(discretization)


    def field(self) -> Field:
        return Field(FieldShape(space=self, components=1))

    @property
    def operators(self) -> DiscreteOperators:
        return self._discretization.operators

    @property
    def bcs(self) -> DiscreteBCs:
        return self._discretization.bcs