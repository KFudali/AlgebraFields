from .space import Space

class SpaceBound():
    def __init__(
        self,
        space: Space
    ):
        self._space = space

    @property
    def space(self) -> Space: return self._space