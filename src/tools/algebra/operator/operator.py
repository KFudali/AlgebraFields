from .core import CoreOperator, binary_ops, unary_ops

class Operator(CoreOperator):
    def __init__(self, input_shape: tuple[int, ...], output_shape: tuple[int, ...]):
        super().__init__(input_shape, output_shape)

    def __add__(self, other: CoreOperator | float):
        if isinstance(other, CoreOperator):
            return binary_ops.AddOperator(self, other)
        if isinstance(other, float):
            return unary_ops.ScalarShiftOperator(self, other)
        return NotImplemented

    def __radd__(self, other: CoreOperator | float):
        if isinstance(other, float):
            return unary_ops.ScalarShiftOperator(self, other)
        return NotImplemented

    def __sub__(self, other: CoreOperator | float):
        if isinstance(other, CoreOperator):
            return binary_ops.SubtractOperator(self, other)
        if isinstance(other, float):
            return unary_ops.ScalarShiftOperator(self, -other)
        return NotImplemented

    def __rsub__(self, other: CoreOperator | float):
        if isinstance(other, float):
            return unary_ops.ScalarShiftOperator(-self, other)
        return NotImplemented

    def __mul__(self, other: CoreOperator | float):
        if isinstance(other, CoreOperator):
            return binary_ops.ElementWiseMulOperator(self, other)
        if isinstance(other, float):
            return unary_ops.ScaleOperator(self, other)
        return NotImplemented

    def __rmul__(self, other: CoreOperator | float):
        if isinstance(other, float):
            return unary_ops.ScaleOperator(self, other)
        return NotImplemented

    def __truediv(self, other: CoreOperator | float):
        if isinstance(other, CoreOperator):
            return binary_ops.ElementWiseMulOperator(self, other)
        if isinstance(other, float):
            return unary_ops.ScaleOperator(self, 1.0 / other)
        return NotImplemented

    def __matmul__(self, other: CoreOperator | float):
        if isinstance(other, CoreOperator):
            return binary_ops.MatMulOperator(self, other)
        return NotImplemented

    def __neg__(self):
        return unary_ops.NegOperator(self)