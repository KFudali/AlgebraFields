from .field import Field, Space

class VectorField(Field):
    def __init__(self, space: Space):
        super().__init__(space)
