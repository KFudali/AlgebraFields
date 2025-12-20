from .field import Field
from ..field_bound import FieldDescriptor
from ..space import Space

class ScalarField(Field):
    def __init__(self, space: Space):
        super().__init__(FieldDescriptor(space, 1))