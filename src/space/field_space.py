import numpy as np
import discr
import tools
from .core import Space
from space.field import Field

class FieldSpace(Space):
    def __init__(self, discretization: discr.Discretization):
        super().__init__(discretization)

    def scalar_field(self, init_value = 0.0) -> Field:
        return self.field(1, init_value)

    def vector_field(self, init_value = 0.0) -> Field:
        return self.field(len(self.shape), init_value)

    def field(self, components: int, init_value = 0.0) -> Field:
        shape = (components, *self.shape)
        value_buffer = tools.buffer.DequeValueBuffer(shape)
        field = Field(self, components, value_buffer)
        init = np.ones(field.shape) * init_value
        field.set_current(init)
        self._time.register_advanceable(field)
        return field
    
