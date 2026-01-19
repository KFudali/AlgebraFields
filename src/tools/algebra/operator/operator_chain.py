import numpy as np

from .core.operator import Operator
from tools.algebra.exceptions import ShapeMismatchException

class OperatorChain(Operator):
    def __init__(self, base_op: Operator):
        super().__init__(base_op.input_shape, base_op.output_shape)
        self._ops = list[Operator]()
        self._ops.append(base_op)

    def append(self, op: Operator):
        if self._ops[-1].output_shape != op.input_shape:
            raise ShapeMismatchException(
                ("Appended operators input_shape does not match previous operator, ",
                f"output_shape.",
                f"Appended: {op.input_shape}, Prev: {self._ops[-1].output_shape}")
            )
        self._ops.append(op)

    def _apply(self, field: np.ndarray, out: np.ndarray):
        self._ops[0].apply(field, out)
        for op in self._ops[1:]:
            new_out = np.zeros(op.output_shape)
            op.apply(out, new_out)