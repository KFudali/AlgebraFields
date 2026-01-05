from .space import Space

class ShapeMismatch(Exception): pass

class FieldShaped():
    def __init__(self, space: Space, components: int):
        self._space = space
        self._components = components

    @property
    def space(self) -> Space:
        return self._space

    @property
    def components(self) -> int:
        return self._components

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.components, *self.space.discretization.shape)

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.components, *self.space.discretization.shape)

    @staticmethod
    def assert_compatible(a: "FieldShaped", b: "FieldShaped", error_msg: str = ""):
        if a.space != b.space:
            raise ShapeMismatch(f"Space mismatch: {error_msg}")
        if a.components != b.components:
            raise ShapeMismatch(f"Component mismatch: {error_msg}")

    def assert_array_shape(self, other: tuple[int, ...], error_msg: str = ""):
        if self.array_shape != other:
            raise ShapeMismatch(
                error_msg + \
                f"\nSelf shape: {self.shape}\n Other shape: {other}."
            )