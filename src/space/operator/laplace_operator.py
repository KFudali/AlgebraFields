import numpy as np

from space.core import Operator, FieldShape


class LaplaceOperator(Operator):
    def __init__(self, shape: FieldShape):
        super().__init__(shape)

    def _apply(self, array: np.ndarray, out: np.ndarray):
        self.space.discretization.operators.laplace(array, out)