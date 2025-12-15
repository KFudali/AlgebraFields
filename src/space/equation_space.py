from discretization import Discretization

from .space import Space
from .boundary import BCFactory
from .field import FieldFactory
from .systems import SystemFactory

class EquationSpace(Space):
    def __init__(
        self, 
        discretization: Discretization
    ):
        super().__init__(discretization)
        self._bc_factory = BCFactory(space = self)
        self._field_factory = FieldFactory(space = self)
        self._system_factory = SystemFactory(space = self)

    @property
    def bcs(self) -> BCFactory: return self._bc_factory
    
    @property
    def fields(self) -> FieldFactory: return self._field_factory
    
    @property
    def systems(self) -> SystemFactory: return self._system_factory