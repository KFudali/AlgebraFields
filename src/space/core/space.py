from model.discrete import Discretization, DiscreteBCs, DiscreteOperators

class Space():
    def __init__(self, discretization: Discretization):
        self._discretization = discretization

    @property
    def discretization(self) -> Discretization:
        return self._discretization
    
    @property
    def shape(self) -> tuple[int, ...]:
        return self._discretization.shape

    @property
    def bcs(self) -> DiscreteBCs:
        return self._discretization.bcs
    
    @property
    def operators(self) -> DiscreteOperators:
        return self._discretization.operators