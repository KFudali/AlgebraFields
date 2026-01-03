from dataclasses import dataclass
from .space import Space

class ShapeMismatch(Exception): pass

@dataclass(frozen=True)
class FieldShape():
    space: Space
    components: int

class ShapeBound():
    def __init__(self, shape: FieldShape):
        self._shape = shape

    @property
    def space(self) -> Space:
        return self._shape.space

    @property
    def components(self) -> int:
        return self._shape.components

    @property
    def shape(self) -> FieldShape: return self._shape

    @property
    def array_shape(self) -> tuple[int, ...]:
        return (self.components, *self.space.discretization.shape)

    @staticmethod
    def assert_compatible(a: "ShapeBound", b: "ShapeBound", error_msg: str = ""):
        if a.space != b.space:
            raise ShapeMismatch(f"Space mismatch: {error_msg}")
        if a.components != b.components:
            raise ShapeMismatch(f"Component mismatch: {error_msg}")

    def assert_array_shape(self, other: tuple[int, ...], error_msg: str = ""):
        if self.array_shape != other:
            raise ShapeMismatch(
                error_msg + \
                f"\nSelf shape: {self.array_shape}\n Other shape: {other}."
            )