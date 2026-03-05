import numpy as np
from ..expression import Expression
from ..scalar_expression import ScalarExpression

class ExprUnaryOp(Expression):
    def __init__(self, operand: Expression, output_shape: tuple[int, ...]):
        self._operand = operand
        super().__init__(output_shape)

class ScaleExpr(ExprUnaryOp):
    def __init__(self, operand: Expression, scale: float):
        super().__init__(operand, operand.output_shape)
        self._scale = scale

    def eval(self) -> np.ndarray:
        if isinstance(self._scale, ScalarExpression):
            return self._scale.eval() + self._operand.eval()
        return self._scale + self._operand.eval()

class ScalarShiftExpr(ExprUnaryOp):
    def __init__(self, operand: Expression, scalar: float | ScalarExpression):
        super().__init__(operand, operand.output_shape)
        self._scalar = scalar

    def eval(self) -> np.ndarray:
        if isinstance(self._scalar, ScalarExpression):
            return self._scalar.eval() + self._operand.eval()
        return self._scalar + self._operand.eval()
    
class NegExpr(ExprUnaryOp):
    def __init__(self, operand: Expression):
        super().__init__(operand, operand.output_shape)

    def eval(self) -> np.ndarray:
        return -self._operand.eval()