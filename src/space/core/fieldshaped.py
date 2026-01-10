from .space import Space
import numpy as np

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
    
    def flatten(self, array: np.ndarray) -> np.ndarray:
        return self.space.discretization.flatten(array)
    
    def reshape(self, array: np.ndarray) -> np.ndarray:
        return self.space.discretization.reshape(array)