import numpy as np
from typing import Callable
from space.base import FieldObject, FieldDescriptor

class FieldValue(FieldObject):
    def __init__(
        self, 
        field_desc: FieldDescriptor,
        value_callable: Callable[[], np.ndarray]
    ):
        super().__init__(field_desc)
        self._value_callable = value_callable

    def eval(self) -> np.ndarray:
        return self._value_callable()