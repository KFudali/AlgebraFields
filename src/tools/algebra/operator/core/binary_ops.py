import numpy as np
from tools.algebra.exceptions import ShapeMismatchException
from .core_operator import CoreOperator

class OperatorBinaryOp(CoreOperator):
    def __init__(
        self, left: CoreOperator, right: CoreOperator, output_shape: tuple[int, ...]
    ):
        self._left = left
        self._right = right
        super().__init__(left.input_shape, output_shape)

class ConstShapeOperatorBinaryOp(OperatorBinaryOp):
    def __init__(self, left: CoreOperator, right: CoreOperator):
        if left.output_shape != right.output_shape:
            raise ShapeMismatchException(
                (
                    "Can only compose expressions of equal shape.",
                    f"left output shape: {left.output_shape}, ",
                    f"right output shape: {right.output_shape}"
                )
            )
        if left.input_shape != right.input_shape:
            raise ShapeMismatchException(
                (
                    "Can only compose expressions of equal shape.",
                    f"left input shape: {left.input_shape}, ",
                    f"right input shape: {right.input_shape}"
                )
            )
        super().__init__(left, right, right.output_shape)

    def apply(self, field: np.ndarray, out: np.ndarray):
        return self._apply(field, out)

class AddOperator(ConstShapeOperatorBinaryOp):
    def __init__(self, left: CoreOperator, right: CoreOperator):
        super().__init__(left, right)

    def _apply(self, field: np.ndarray, out: np.ndarray):
        right_out = np.zeros_like(out)
        self._right.apply(field, right_out)
        self._left.apply(field, out)
        out[:] += right_out[:]

class SubtractOperator(ConstShapeOperatorBinaryOp):
    def __init__(self, left: CoreOperator, right: CoreOperator):
        super().__init__(left, right)

    def _apply(self, field: np.ndarray, out: np.ndarray):
        right_out = np.zeros_like(out)
        self._right.apply(field, right_out)
        self._left.apply(field, out)
        out[:] -= right_out[:]

class ElementWiseMulOperator(ConstShapeOperatorBinaryOp):
    def __init__(self, left: CoreOperator, right: CoreOperator):
        super().__init__(left, right, left.output_shape)

    def _apply(self, field: np.ndarray, out: np.ndarray):
        right_out = np.zeros_like(out)
        self._right.apply(field, right_out)
        self._left.apply(field, out)
        out[:] *= right_out[:]

class ElementWiseDivOperator(ConstShapeOperatorBinaryOp):
    def __init__(self, left: CoreOperator, right: CoreOperator):
        super().__init__(left, right, left.output_shape)

    def _apply(self, field: np.ndarray, out: np.ndarray):
        right_out = np.zeros_like(out)
        self._right.apply(field, right_out)
        self._left.apply(field, out)
        out[:] /= right_out[:]


def matmul_shape(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    if len(a) == 0 or len(b) == 0:
        raise ShapeMismatchException("Scalars not supported in matmul")

    # Vector @ vector
    if len(a) == 1 and len(b) == 1:
        if a[0] != b[0]:
            raise ShapeMismatchException("Vector dot requires same length")
        return ()

    # Vector @ matrix
    if len(a) == 1:
        if a[0] != b[-2]:
            raise ShapeMismatchException("Incompatible shapes for matmul")
        return b[:-2] + (b[-1],)

    # Matrix @ vector
    if len(b) == 1:
        if a[-1] != b[0]:
            raise ShapeMismatchException("Incompatible shapes for matmul")
        return a[:-2] + (a[-2],)

    # General case (..., m, k) @ (..., k, n)
    if a[-1] != b[-2]:
        raise ShapeMismatchException("Incompatible shapes for matmul")

    # Broadcast batch dimensions
    batch_a = a[:-2]
    batch_b = b[:-2]

    if len(batch_a) != len(batch_b):
        raise ShapeMismatchException("Batch dimensions not aligned")

    batch = []
    for da, db in zip(batch_a, batch_b):
        if da == db:
            batch.append(da)
        else:
            raise ShapeMismatchException("Batch dimensions must match exactly")

    return tuple(batch) + (a[-2], b[-1])


class MatMulOperator(OperatorBinaryOp):
    def __init__(self, left: CoreOperator, right: CoreOperator):
        output_shape = matmul_shape(left.output_shape, right.output_shape)
        super().__init__(left, right, output_shape)

    def _apply(self, field: np.ndarray, out: np.ndarray):
        NotImplemented

        # right_out = np.zeros_like(out)
        # self._right.apply(field, right_out)
        # self._left.apply(field, out)
        # out