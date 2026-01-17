import numpy as np
from .core_operator import CoreOperator

class OperatorUnaryOp(CoreOperator):
    def __init__(self, operand: CoreOperator, output_shape: tuple[int, ...]):
        self._operand = operand
        super().__init__(operand.input_shape, output_shape)

class ScaleOperator(OperatorUnaryOp):
    def __init__(self, operand: CoreOperator, scale: float):
        super().__init__(operand, operand.output_shape)
        self._scale = scale

    def _apply(self, field: np.ndarray, out: np.ndarray) -> np.ndarray:
        self._operand.apply(field, out)
        out[:] *= self._scale

class ScalarShiftOperator(OperatorUnaryOp):
    def __init__(self, operand: CoreOperator, scalar: float):
        super().__init__(operand, operand.output_shape)
        self._scalar = scalar

    def _apply(self, field: np.ndarray, out: np.ndarray) -> np.ndarray:
        self._operand.apply(field, out)
        out[:] += self._scalar
    
class NegOperator(OperatorUnaryOp):
    def __init__(self, operand: CoreOperator):
        super().__init__(operand, operand.output_shape)

    def _apply(self, field: np.ndarray, out: np.ndarray) -> np.ndarray:
        self._operand.apply(field, out)
        out[:] = -out[:]