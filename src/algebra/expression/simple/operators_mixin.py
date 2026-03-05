from .simple_expr import Expression
from ..scalar_expression import ScalarExpression


class ExpressionOperatorsMixin:
    def _make_unary(self, op: Expression) -> Expression:
        return op

    def _make_binary(self, op: Expression) -> Expression:
        return op

    def __add__(self, other: Expression | ScalarExpression | float):
        from .binary_ops import AddExpr
        from .unary_ops import ScalarShiftExpr

        if isinstance(other, Expression):
            return self._make_binary(AddExpr(self, other))
        if isinstance(other, (float, ScalarExpression)):
            return self._make_unary(ScalarShiftExpr(self, other))
        return NotImplemented

    def __mul__(self, other):
        from .binary_ops import ElementWiseMulExpr
        from .unary_ops import ScaleExpr

        if isinstance(other, Expression):
            return self._make_binary(ElementWiseMulExpr(self, other))
        if isinstance(other, (float, ScalarExpression)):
            return self._make_unary(ScaleExpr(self, other))
        return NotImplemented

    def __truediv__(self, other):
        from .binary_ops import ElementWiseDivExpr
        from .unary_ops import ScaleExpr

        if isinstance(other, Expression):
            return self._make_binary(ElementWiseDivExpr(self, other))
        if isinstance(other, (float, ScalarExpression)):
            return self._make_unary(ScaleExpr(self, 1.0 / other))
        return NotImplemented

    def __neg__(self):
        from .unary_ops import NegExpr
        return self._make_unary(NegExpr(self))