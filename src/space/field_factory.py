from .base import SpaceBound, Space
from .field import ScalarField, VectorField

class FieldFactory(SpaceBound):
    def __init__(
        self, 
        space: Space
    ):
        super().__init__(space)
    
    def scalar(self) -> ScalarField: pass
    
    def vector(self) -> VectorField: pass