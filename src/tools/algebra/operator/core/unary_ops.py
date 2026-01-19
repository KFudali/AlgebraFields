import numpy as np
from .operator import Operator

class OperatorUnaryOp(Operator):
    def __init__(self, operand: Operator, output_shape: tuple[int, ...]):
        self._operand = operand
        super().__init__(operand.input_shape, output_shape)

class OperatorScaleOp(OperatorUnaryOp):
    def __init__(self, operand: Operator, scale: float):
        super().__init__(operand, operand.output_shape)
        self._scale = scale

    def _apply(self, field: np.ndarray, out: np.ndarray) -> np.ndarray:
        self._operand.apply(field, out)
        out[:] *= self._scale

class OperatorScalarShiftOp(OperatorUnaryOp):
    def __init__(self, operand: Operator, scalar: float):
        super().__init__(operand, operand.output_shape)
        self._scalar = scalar

    def _apply(self, field: np.ndarray, out: np.ndarray) -> np.ndarray:
        self._operand.apply(field, out)
        out[:] += self._scalar
    
class OperatorNegOp(OperatorUnaryOp):
    def __init__(self, operand: Operator):
        super().__init__(operand, operand.output_shape)

    def _apply(self, field: np.ndarray, out: np.ndarray) -> np.ndarray:
        self._operand.apply(field, out)
        out[:] = -out[:]