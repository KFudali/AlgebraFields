import numpy as np
from .core_expression import CoreExpression

class ExprUnaryOp(CoreExpression):
    def __init__(self, operand: CoreExpression, output_shape: tuple[int, ...]):
        self._operand = operand
        super().__init__(output_shape)

class ScaleExpr(ExprUnaryOp):
    def __init__(self, operand: CoreExpression, scale: float):
        super().__init__(operand, operand.output_shape)
        self._scale = scale

    def eval(self) -> np.ndarray:
        return self._scale * self._operand.eval()

class ScalarShiftExpr(ExprUnaryOp):
    def __init__(self, operand: CoreExpression, scalar: float):
        super().__init__(operand, operand.output_shape)
        self._scalar = scalar

    def eval(self) -> np.ndarray:
        return self._scalar * self._operand.eval()
    
class NegExpr(ExprUnaryOp):
    def __init__(self, operand: CoreExpression):
        super().__init__(operand, operand.output_shape)

    def eval(self) -> np.ndarray:
        return -self._operand.eval()