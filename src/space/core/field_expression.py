
from typing import Callable
import numpy as np

from tools.algebra.expr import Expression, CallableExpression
from .fieldshaped import FieldShaped, Space

class FieldExpression(Expression, FieldShaped):
    def __init__(self, space: Space, components: int):
        FieldShaped.__init__(self, space, components)
        Expression.__init__(self, self.shape)

class CallableFieldExpression(CallableExpression, FieldShaped):
    def __init__(
        self, 
        space: Space, 
        components: int, 
        expr: Callable[[], np.ndarray]
    ):
        FieldShaped.__init__(self, space, components)
        CallableExpression.__init__(self, self.shape, expr)