import numpy as np

from .expression import Expression
from ..operator import Operator

class ApplyOperatorExpr(Expression):
    def __init__(self, op: Operator, expr: Expression):
        super().__init__(op.output_shape)
        self._op = op
        self._expr = expr

    @property
    def operator(self) -> Operator: return self._op

    @property
    def expr(self) -> Expression: return self._expr

    def eval(self) -> np.ndarray:
        x = self._expr.eval()
        out = np.zeros_like(x)
        self._op.apply(x, out)
        return out