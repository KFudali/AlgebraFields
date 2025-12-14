from discretization import Discretization

from .space import Space
from .boundary import BCFactory
from .field import FieldFactory

class EquationSpace(Space):
    def __init__(
        self, 
        discretization: Discretization
    ):
        super().__init__(discretization)
        self._bc_factory = BCFactory(space = self)
        self._field_factory = FieldFactory(space = self)

    @property
    def bcs(self) -> BCFactory: return self._bc_factory
    
    @property
    def fields(self) -> FieldFactory: return self._field_factory