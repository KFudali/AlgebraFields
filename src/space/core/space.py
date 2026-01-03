from model.discretization import Discretization

class Space():
    def __init__(self, discretization: Discretization):
        self._discretization = discretization
    
    @property
    def discretization(self) -> Discretization:
        return self._discretization
    
    @property
    def dim(self) -> int: 
        return self._discretization.dim