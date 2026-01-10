from typing import Callable
import numpy as np

from tools.algebra.operator import Operator, CallableOperator
from .fieldshaped import FieldShaped, Space

class FieldOperator(Operator, FieldShaped):
    def __init__(self, space: Space, components: int):
        FieldShaped.__init__(self, space, components)
        Operator.__init__(self, self.shape, self.shape)

class CallableFieldOperator(CallableOperator, FieldShaped):
    def __init__(
        self, 
        space: Space, 
        components: int, 
        apply: Callable[[np.ndarray, np.ndarray], None]
    ):
        FieldShaped.__init__(self, space, components)
        CallableOperator.__init__(self, self.shape, self.shape, apply)
