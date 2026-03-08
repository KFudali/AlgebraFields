from abc import ABC, abstractmethod

from .boundary_id import BoundaryId
from .boundary import Boundary

class Domain(ABC):
    @property
    @abstractmethod
    def boundaries(self) -> list[BoundaryId]: pass
    
    @abstractmethod
    def boundary(self, boundary_id: BoundaryId) -> Boundary: pass