from space.base import FieldDescriptor,  Space
from .field import Field
from .steady_value_buffer import SteadyValueBuffer

class VectorField(Field):
    def __init__(
        self, space: Space
    ):
        super().__init__(
            SteadyValueBuffer(
                FieldDescriptor(space, components=space.dim)
            )
        )