import numpy as np

from .linear_operator import LinearOperator
from space.base import AbstractField

class GradientOperator(LinearOperator):
    def __init__(self, field: AbstractField):
        super().__init__(field)

    @property
    def stencil(self) -> np.ndarray:
        return self._field.disc.stencils.grad()
    
    def eval(self):
        return np.dot(self.stencil, self._field.value())