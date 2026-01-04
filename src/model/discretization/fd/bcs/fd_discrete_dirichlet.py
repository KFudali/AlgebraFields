import numpy as np

from model.core import DiscreteDirichletBC
from ..domain import FDBoundary
from algebra.operator import Operator, CallableOperator

class FDDirichletBC(DiscreteDirichletBC):
    def __init__(self, boundary: FDBoundary, value: float):
        super().__init__()
        self._value = value
        self._boundary = boundary

    def apply_linear(self, op: Operator, b: np.ndarray) -> Operator:
        ids = self._boundary.ids
        def modified_apply(field: np.ndarray, out: np.ndarray):
            op.apply(field, out)
            out[ids, :] = 0.0
            out[ids, ids] = 1.0
        b[ids] = self._value
        return CallableOperator(op.input_shape, op.output_shape, modified_apply)