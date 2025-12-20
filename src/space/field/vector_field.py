from .field import Field
from ..field_bound import FieldDescriptor
from ..space import Space

class ScalarField(Field):
    def __init__(self, space: Space):
        span = space.shape[0]
        super().__init__(FieldDescriptor(space, span))