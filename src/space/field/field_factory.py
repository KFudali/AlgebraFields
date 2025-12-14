from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..space_bound import SpaceBound, Space

from .scalar_field import ScalarField
from .vector_field import VectorField

class FieldFactory(SpaceBound):
    def __init__(
        self, 
        space: Space
    ):
        super().__init__(space)
    
    def scalar(self) -> ScalarField: pass
    
    def vector(self) -> VectorField: pass