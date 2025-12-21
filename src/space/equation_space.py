from model.discretization import Discretization
from .base import Space, bc

from .field_factory import FieldFactory

class EquationSpace(Space):
    def __init__(
        self, 
        discretization: Discretization
    ):
        self._discretization = discretization
        self._field_factory = FieldFactory(space = self)
        self._bc_factory = bc.BCFactory(space = self)

    @property
    def disc(self) -> Discretization:
        return self._discretization
    
    @property
    def bcs(self) -> bc.BCFactory: return self._bc_factory
    
    @property
    def fields(self) -> FieldFactory: return self._field_factory