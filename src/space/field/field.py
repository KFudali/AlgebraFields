from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..space_bound import SpaceBound, Space

from ..boundary import BC

class Field(SpaceBound, ABC):
    def __init__(
        self,
        space: Space
    ):
        SpaceBound.__init__(space = space)
    
    @abstractmethod
    def apply_bc(self, boundary_condition: BC): pass