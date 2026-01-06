from abc import abstractmethod, ABC
from typing import Generic
from algebra.system import LES

from model.domain import BoundaryType


class DiscreteBC(ABC, Generic[BoundaryType]):
    def __init__(self, boundary: BoundaryType):
        super().__init__()
        self._boundary = boundary

    @property
    def boundary(self) -> BoundaryType:
        return self._boundary

    @abstractmethod 
    def apply_to_system(self, system: LES): pass
