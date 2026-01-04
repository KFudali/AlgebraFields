from model.discretization import Discretization, DiscreteBCs

class Space():
    def __init__(self, discretization: Discretization):
        self._discretization = discretization
    
    @property
    def discretization(self) -> Discretization:
        return self._discretization
    