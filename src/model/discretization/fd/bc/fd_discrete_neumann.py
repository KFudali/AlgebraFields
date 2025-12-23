
import numpy as np

from ...boundaries import DiscreteNeumannBC
from ..domain import FDBoundary

class FDNeumannBC(DiscreteNeumannBC):
    def __init__(self):
        super().__init__()

    def apply_linear(
        A: np.ndarray, b: np.ndarray, value: float, boundary: FDBoundary
    ):
        ids = boundary.ids
        A[ids, :] = 0.0
        A[:, ids] = 0.0
        A[ids, ids] = 1.0
        b[ids] = value