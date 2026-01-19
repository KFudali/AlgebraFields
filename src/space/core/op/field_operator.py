from __future__ import annotations
from typing import Callable, Self
import numpy as np

from tools.algebra.operator.core import Operator, unary_ops, binary_ops
from ..fieldshaped import FieldShaped, Space
import numpy as np


class FieldOperator(Operator, FieldShaped):
    def __init__(self, space: Space, components: int):
        FieldShaped.__init__(self, space, components)
        Operator.__init__(self, self.shape, self.shape)


    def _make_unary(self, op: unary_ops.OperatorUnaryOp) -> Self:
        operator = super()._make_unary(op)
        from .field_operator_wrapper import FieldOperatorWrapper
        return FieldOperatorWrapper(self.space, self.components, operator)

    def _make_binary(self, op: binary_ops.OperatorBinaryOp) -> Self:
        operator = super()._make_binary(op)
        from .field_operator_wrapper import FieldOperatorWrapper
        return FieldOperatorWrapper(self.space, self.components, operator)
    
class CallableFieldOperator(FieldOperator):
    def __init__(
        self, 
        space: Space, 
        components: int, 
        apply: Callable[[np.ndarray, np.ndarray], None]
    ):
        FieldOperator.__init__(self, space, components)
        self._apply_callable = apply

    def _apply(self, field: np.ndarray, out: np.ndarray):
        self._apply_callable(field, out)