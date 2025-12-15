import numpy as np

from .linear_operator import LinearOperator
from space.field import Field

class LaplaceOperator(LinearOperator):
    def __init__(self, field: Field):
        super().__init__(field)

    @property
    def stencil(self) -> np.ndarray:
        return self._field.discretization.stencils.laplace()
    
    def eval(self):
        return np.dot(self.stencil, self._field.value())