from .core import Space
from model.discretization import Discretization, DiscreteBCs



class EquationSpace(Space):
    def __init__(self, discretization: Discretization):
        super().__init__(discretization)

    @property
    def bcs(self) -> DiscreteBCs:
        return self._discretization.bcs