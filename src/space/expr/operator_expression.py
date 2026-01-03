import numpy as np

from space.core.shapebound import ShapeBound
from space.core.operator import Operator
from space.core.expression import Expression

class OperatorExpression(Expression):
    def __init__(self, op: Operator, expr: Expression):
        ShapeBound.assert_compatible(op, expr)
        super().__init__(op.shape)
        self._op = op
        self._expr = expr

    def _eval(self) -> np.ndarray:
        x = self._expr.eval()
        out = np.zeros_like(x)
        self._op.apply(x, out)
        return out