import numpy as np


from ...boundaries import DiscreteDirichletBC
from ..domain import FDBoundary

class FDDirichletBC(DiscreteDirichletBC):
    def __init__(self):
        super().__init__()

    def apply_linear(
        self, A: np.ndarray, b: np.ndarray, value: float, boundary: FDBoundary
    ):
        ids = boundary.ids
        A[ids, :] = 0.0
        A[ids, ids] = 1.0
        b[ids] = value