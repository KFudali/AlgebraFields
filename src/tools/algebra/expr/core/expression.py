from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Type, Self, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .binary_ops import ExprBinaryOp
    from .unary_ops import ExprUnaryOp

class Expression(ABC):
    def __init__(self, output_shape: tuple[int, ...]):
        self._output_shape = output_shape

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self._output_shape

    @abstractmethod
    def eval(self) -> np.ndarray:
        pass

    def _make_unary(self, op: ExprUnaryOp) -> Self: 
        return op

    def _make_binary(self, op: ExprBinaryOp) -> Self:
        return op

    @staticmethod 
    def _neg_expr(op: Expression) -> Self:
        from .unary_ops import NegExpr
        return NegExpr(op)

    def __add__(self, other: "Expression" | float):
        from .binary_ops import AddExpr
        from .unary_ops import ScalarShiftExpr
        if isinstance(other, Expression):
            return self._make_binary(AddExpr(self, other))
        if isinstance(other, float):
            return self._make_unary(ScalarShiftExpr(self, other))
        return NotImplemented

    def __radd__(self, other: float):
        from .unary_ops import ScalarShiftExpr
        if isinstance(other, float):
            return self._make_binary(ScalarShiftExpr(self, other))
        return NotImplemented

    def __sub__(self, other: "Expression" | float):
        from .binary_ops import SubtractExpr
        from .unary_ops import ScalarShiftExpr

        if isinstance(other, Expression):
            return self._make_binary(SubtractExpr(self, -other))
        if isinstance(other, float):
            return self._make_unary(ScalarShiftExpr(self, -other))
        return NotImplemented

    def __rsub__(self, other: float):
        from .unary_ops import ScalarShiftExpr
        if isinstance(other, float):
            neg_self = self._neg_expr(self)
            return self._make_unary(ScalarShiftExpr(neg_self, other))
        return NotImplemented

    def __mul__(self, other: "Expression" | float):
        from .binary_ops import ElementWiseMulExpr
        from .unary_ops import ScaleExpr

        if isinstance(other, Expression):
            return self._make_binary(ElementWiseMulExpr(self, other))
        if isinstance(other, float):
            return self._make_unary(ScaleExpr(self, other))
        return NotImplemented

    def __rmul__(self, other: float):
        from .unary_ops import ScaleExpr
        if isinstance(other, float):
            return self._make_unary(ScaleExpr(self, other))
        return NotImplemented

    def __truediv__(self, other: "Expression" | float):
        from .binary_ops import ElementWiseDivExpr
        from .unary_ops import ScaleExpr

        if isinstance(other, Expression):
            return self._make_binary(ElementWiseDivExpr(self, other))
        if isinstance(other, float):
            return self._make_unary(ScaleExpr(self, 1.0 / other))
        return NotImplemented

    def __matmul__(self, other: "Expression"):
        from .binary_ops import MatMulExpr

        if isinstance(other, Expression):
            return self._make_binary(MatMulExpr(self, other))
        return NotImplemented

    def __neg__(self):
        return self._neg_expr(self)