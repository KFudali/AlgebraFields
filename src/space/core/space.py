from discretization.core import Discretization, DiscreteBCFactory, DiscreteOperatorsFactory

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
    def bcs(self) -> DiscreteBCFactory:
        return self._discretization.bcs
    
    @property
    def operators(self) -> DiscreteOperatorsFactory:
        return self._discretization.operators