
from space.base import FieldDescriptor,  Space
from .field import Field

class ScalarField(Field):
    def __init__(
        self, space: Space
    ):
        super().__init__(FieldDescriptor(space, components=1))