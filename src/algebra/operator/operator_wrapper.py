from typing import Self
import numpy as np
from .operator import Operator

class OperatorWrapper(Operator):
    def __init__(
        self, 
        operator: Operator, 
        input_shape: tuple[int, ...] | None = None,
        output_shape: tuple[int, ...] | None = None
    ):
        if input_shape == None:
            input_shape = operator.input_shape
        if output_shape == None:
            output_shape = operator.output_shape
        super().__init__(input_shape, output_shape)
        self._op = operator

    @property
    def core(self) -> Operator:
        return self._op

    def copy(self) -> "OperatorWrapper":
        return self._new(self._op.copy())

    def _new(self, operator: Operator) -> Self:
        return OperatorWrapper(operator, self.input_shape, self.output_shape)

    def _apply(self, field: np.ndarray, out: np.ndarray):
        return self._op._apply(field, out)

    def __neg__(self) -> Self:
        return OperatorWrapper(-self.core, self.input_shape, self.output_shape)

    def __add__(self, other) -> Self:
        result = self._op.__add__(other)
        if result is not NotImplemented: 
            return self._new(result)
        
        result = other.__radd__(self._op)
        if result is NotImplemented: return NotImplemented
        return result

    def __radd__(self, other) -> Self:
        result = self._op.__add__(other)
        if result is NotImplemented: return NotImplemented
        return self._new(result)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other) -> Self:
        result = self._op.__mul__(other)
        if result is NotImplemented: return NotImplemented
        return self._new(result)

    def __rmul__(self, other) -> Self:
        result = self._op.__rmlu__(other)
        if result is NotImplemented: return NotImplemented
        return self._new(result)

    def __truediv__(self, other) -> Self:
        result = self._op.__truediv__(other)
        if result is NotImplemented: return NotImplemented
        return self._new(result)