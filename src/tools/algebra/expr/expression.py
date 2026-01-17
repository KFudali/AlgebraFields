from .core import CoreExpression
from .core import binary_ops, unary_ops

class Expression(CoreExpression):
    def __init__(self, output_shape: tuple[int, ...]):
        super().__init__(output_shape)

    def __add__(self, other: CoreExpression | float):
        if isinstance(other, CoreExpression):
            return binary_ops.AddExpr(self, other)
        if isinstance(other, float):
            return unary_ops.ScalarShiftExpr(self, other)
        return NotImplemented

    def __radd__(self, other: CoreExpression | float):
        if isinstance(other, float):
            return unary_ops.ScalarShiftExpr(self, other)
        return NotImplemented

    def __sub__(self, other: CoreExpression | float):
        if isinstance(other, CoreExpression):
            return binary_ops.SubtractExpr(self, other)
        if isinstance(other, float):
            return unary_ops.ScalarShiftExpr(self, -other)
        return NotImplemented

    def __rsub__(self, other: CoreExpression | float):
        if isinstance(other, float):
            return unary_ops.ScalarShiftExpr(-self, other)
        return NotImplemented

    def __mul__(self, other: CoreExpression | float):
        if isinstance(other, CoreExpression):
            return binary_ops.ElementWiseMulExpr(self, other)
        if isinstance(other, float):
            return unary_ops.ScaleExpr(self, other)
        return NotImplemented

    def __truediv__(self, other: CoreExpression | float):
        if isinstance(other, CoreExpression):
            return binary_ops.ElementWiseDivExpr(self, other)
        if isinstance(other, float):
            return unary_ops.ScaleExpr(self, 1.0 / other)
        return NotImplemented

    def __rmul__(self, other: CoreExpression | float):
        if isinstance(other, float):
            return unary_ops.ScaleExpr(self, other)
        return NotImplemented

    def __matmul__(self, other: CoreExpression | float):
        if isinstance(other, CoreExpression):
            return binary_ops.MatMulExpr(self, other)
        return NotImplemented
    

    def __neg__(self):
        return unary_ops.NegExpr(self)