from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Type, Self, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .binary_ops import OperatorBinaryOp
    from .unary_ops import OperatorUnaryOp
from tools.algebra.exceptions import ShapeMismatchException


class Operator(ABC):
    def __init__(self, input_shape: tuple[int, ...], output_shape: tuple[int, ...]):
        self._input_shape = input_shape
        self._output_shape = output_shape

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self._input_shape

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self._output_shape

    def apply(self, field: np.ndarray, out: np.ndarray):
        if field.shape != self.input_shape:
            raise ShapeMismatchException(
                "Cannot apply operator. Input size differs from operator size. "
                f"Output shape: {field.shape}, ",
                f"Operator input shape: {self.input_shape}"
            )
        if out.shape != self.output_shape:
            raise ShapeMismatchException(
                "Cannot apply operator. Output size differs from operator size. "
                f"Output shape: {out.shape}, ",
                f"Operator output shape: {self.output_shape}"
            )
        self._apply(field, out)

    @abstractmethod
    def _apply(self, field: np.ndarray, out: np.ndarray): pass



    def _make_unary(self, op: OperatorUnaryOp) -> Self:
        return op

    def _make_binary(self, op: OperatorBinaryOp) -> Self:
        return op 

    def __neg__(self) -> Self:
        from .unary_ops import OperatorNegOp
        return self._make_unary(OperatorNegOp(self))

    def __add__(self, other: Operator | float) -> Self:
        from .binary_ops import OperatorAddOp
        from .unary_ops import OperatorScalarShiftOp

        if isinstance(other, Operator):
            return self._make_binary(OperatorAddOp(self, other))
        if isinstance(other, float):
            return self._make_unary(OperatorScalarShiftOp(self, other))
        return NotImplemented

    def __radd__(self, other: float) -> Self:
        from .unary_ops import OperatorScalarShiftOp

        if isinstance(other, float):
            return self._make_unary(OperatorScalarShiftOp(self, other))
        return NotImplemented

    def __sub__(self, other: Operator | float) -> Self:
        from .binary_ops import OperatorSubtractOp
        from .unary_ops import OperatorScalarShiftOp

        if isinstance(other, Operator):
            return self._make_binary(OperatorSubtractOp(self, other))
        if isinstance(other, float):
            return self._make_unary(OperatorScalarShiftOp(self, -other))
        return NotImplemented

    def __rsub__(self, other: float) -> Self:
        from .unary_ops import OperatorScalarShiftOp, OperatorNegOp

        if isinstance(other, float):
            return self._make_unary(OperatorScalarShiftOp(OperatorNegOp(self), other))
        return NotImplemented

    def __mul__(self, other: Operator | float) -> Self:
        from .binary_ops import OperatorElementWiseDivOp
        from .unary_ops import OperatorScaleOp

        if isinstance(other, Operator):
            return self._make_binary(OperatorElementWiseDivOp(self, other))
        if isinstance(other, float):
            return self._make_unary(OperatorScaleOp(self, other))
        return NotImplemented

    def __rmul__(self, other: float) -> Self:
        from .unary_ops import OperatorScaleOp

        if isinstance(other, float):
            return self._make_unary(OperatorScaleOp(self, other))
        return NotImplemented

    def __truediv__(self, other: Operator | float) -> Self:
        from .binary_ops import OperatorElementWiseDivOp
        from .unary_ops import OperatorScaleOp

        if isinstance(other, Operator):
            return self._make_binary(OperatorElementWiseDivOp(self, other))
        if isinstance(other, float):
            return self._make_unary(OperatorScaleOp(self, 1.0 / other))
        return NotImplemented

    def __matmul__(self, other: Operator) -> Self:
        from .binary_ops import OperatorMatMulOp

        if isinstance(other, Operator):
            return self._make_binary(OperatorMatMulOp(self, other))
        return NotImplemented
