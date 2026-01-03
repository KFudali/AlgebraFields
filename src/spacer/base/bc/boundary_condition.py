from abc import abstractmethod
import numpy as np

from model.domain.boundary import Boundary
from spacer.base import SpaceBound, Space

class BoundaryCondition(SpaceBound):
    def __init__(
        self,
        space: Space,
        boundary: Boundary
    ):
        super().__init__(space)
        self._boundary = boundary
    
    @abstractmethod
    def apply_linear(self, A: np.ndarray, b: np.ndarray): pass

BC = BoundaryCondition