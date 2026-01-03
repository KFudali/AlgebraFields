import numpy as np
from spacer.base import FieldValueBuffer, FieldDescriptor

class SteadyValueBuffer(FieldValueBuffer):
    def __init__(self, field_descriptor: FieldDescriptor):
        super().__init__(field_descriptor)
        components = field_descriptor.components
        discretization_shape = field_descriptor.space.disc.shape
        self._value = np.zeros(shape=(components, discretization_shape))

    def get(self) -> np.ndarray:
        return self._value

    def set(self, value: np.ndarray):
        if self.desc.shape != value.shape: raise RuntimeError()
        self._value = value