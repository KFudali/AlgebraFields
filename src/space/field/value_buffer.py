import numpy as np


class ValueBuffer():
    def __init__(self, shape: tuple[int, ...]):
        self._saved_steps = 1
        self._shape = shape
        self._values = list[np.ndarray]()

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def saved_steps(self) -> int: return self._saved_steps

    def set_saved_steps(self, steps: int):
        if steps > self.saved_steps:
            self._saved_steps = steps
    
    def get(self, past_step: int = 0) -> np.ndarray:
        if past_step >= len(self._values):
            raise RuntimeError("Past not available")
        return self._values[past_step]
    
    def set_current(self, value: np.ndarray):
        if value.shape != self.shape:
            raise RuntimeError("Shape mismatch")
        self._values[0] = value
    
    def advance(self):
        if len(self._values) == self.saved_steps:
            self._values.pop()
        self._values.insert(0, np.zeros(self._shape))

