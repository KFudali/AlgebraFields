import numpy as np
from typing import TYPE_CHECKING, Callable
if TYPE_CHECKING:
    from ..field_bound import FieldBound, FieldDescriptor

class FieldValue(FieldBound):
    def __init__(
        self, 
        field_desc: FieldDescriptor,
        value_callable: Callable[[], np.ndarray]
    ):
        super().__init__(field_desc)
        self._value_callable = value_callable

    def eval(self) -> np.ndarray:
        return self._value_callable()