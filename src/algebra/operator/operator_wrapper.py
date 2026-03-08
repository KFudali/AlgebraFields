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

    def _wrap_magic(self, method, other):
        return self._new(method(other))

    def __neg__(self) -> Self:
        return OperatorWrapper(-self.core, self.input_shape, self.output_shape)

    def __add__(self, other) -> Self:
        return self._wrap_magic(self._op.__add__, other)

    def __mul__(self, other) -> Self:
        return self._wrap_magic(self._op.__mul__, other)

    def __truediv__(self, other) -> Self:
        return self._wrap_magic(self._op.__truediv__, other)