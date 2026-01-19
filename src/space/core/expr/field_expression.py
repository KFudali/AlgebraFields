from __future__ import annotations
from typing import Self, Callable
import numpy as np

from tools.algebra.exceptions import ShapeMismatchException
from tools.algebra.expr.core import Expression, unary_ops, binary_ops
from ..fieldshaped import FieldShaped, Space

class FieldExpression(Expression, FieldShaped):
    def __init__(self, space: Space, components: int):
        FieldShaped.__init__(self, space, components)
        Expression.__init__(self, self.shape)
    
    def _make_unary(self, op: unary_ops.ExprUnaryOp) -> Self:
        from .field_expression_wrapper import FieldExpressionWrapper
        expr = super()._make_unary(op)
        return FieldExpressionWrapper(self.space, self.components, expr)
    
    def _make_binary(self, op: binary_ops.ExprBinaryOp) -> Self:
        from .field_expression_wrapper import FieldExpressionWrapper
        expr = super()._make_binary(op)
        return FieldExpressionWrapper(self.space, self.components, expr)
    

class CallableFieldExpression(FieldExpression):
    def __init__(
        self, 
        space: Space, 
        components: int, 
        expr: Callable[[], np.ndarray]
    ):
        super().__init__(space, components)
        self._expr = expr

    def eval(self) -> np.ndarray:
        value = self._expr()
        if value.shape != self.output_shape:
            raise ShapeMismatchException(
                f"Callable shape: {value.shape}. Expr shape: {self.output_shape}."
            )
        return value