from abc import abstractmethod
import numpy as np

from model.domain.boundary import Boundary
from ..shapebound import ShapeBound, FieldShape
from ..operator import Operator

class BoundaryCondition(ShapeBound):
    def __init__(
        self, shape: FieldShape, boundary: Boundary
    ):
        super().__init__(shape)
        self._boundary = boundary
    
    @abstractmethod
    def apply_linear(
        self, A: Operator, b: np.ndarray
    ) -> Operator: pass

BC = BoundaryCondition