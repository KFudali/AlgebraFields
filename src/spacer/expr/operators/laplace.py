import numpy as np

from .linear_operator import LinearOperator
from spacer.base import AbstractField

class LaplaceOperator(LinearOperator):
    def __init__(self, field: AbstractField):
        super().__init__(field)

    @property
    def stencil(self) -> np.ndarray:
        return self._field.disc.stencils.laplace()
    
    def eval(self):
        return np.dot(self.stencil, self._field.raw_value())