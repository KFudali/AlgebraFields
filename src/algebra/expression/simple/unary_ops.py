import numpy as np
from .simple_expr import SimpleExpression, Expression
from ..scalar_expression import ScalarExpression

class ExprUnaryOp(SimpleExpression):
    def __init__(self, operand: Expression, output_shape: tuple[int, ...]):
        self._operand = operand
        super().__init__(output_shape)

class ScaleExpr(ExprUnaryOp):
    def __init__(self, operand: Expression, scale: float):
        super().__init__(operand, operand.output_shape)
        self._scale = ScalarExpression.ensure(scale)

    def eval(self) -> np.ndarray:
        return self._operand.eval() / self._scale.eval()

class ScalarShiftExpr(ExprUnaryOp):
    def __init__(self, operand: Expression, scalar: float | ScalarExpression):
        super().__init__(operand, operand.output_shape)
        self._scalar = ScalarExpression.ensure(scalar)

    def eval(self) -> np.ndarray:
        return  self._operand.eval() + self._scalar
    
class NegExpr(ExprUnaryOp):
    def __init__(self, operand: Expression):
        super().__init__(operand, operand.output_shape)

    def eval(self) -> np.ndarray:
        return -self._operand.eval()