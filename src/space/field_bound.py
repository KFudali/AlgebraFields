from dataclasses import dataclass
from .space import Space
from .space_bound import SpaceBound

@dataclass
class FieldDescriptor():
    space: Space
    components: int = 1

class FieldBound(SpaceBound):
    def __init__(self, field_descriptor: FieldDescriptor):
        self._field_descriptor = field_descriptor

    def same_shape(self, other: "FieldBound"):
        if self.desc == other.desc: return True
        else: return False

    @property
    def desc(self) -> FieldDescriptor: return self._field_descriptor

    @property
    def space(self) -> Space: return self._field_descriptor.space

    @property
    def components(self) -> int: return self._field_descriptor.components

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.components, self.space.shape)