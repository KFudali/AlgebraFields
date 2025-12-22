from .base import SpaceBound, Space
from .field import ScalarField, VectorField

class FieldFactory(SpaceBound):
    def __init__(
        self, 
        space: Space
    ):
        super().__init__(space)
    
    def scalar(self) -> ScalarField:
        return ScalarField(self._space)

    def vector(self) -> VectorField:
        return VectorField(self._space)