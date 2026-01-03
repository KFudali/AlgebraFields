import numpy as np
from spacer.base import FieldValueBuffer, FieldDescriptor
from .history_buffer import HistoryBuffer
from spacer.time import TimeStep

class TransientValueBuffer(FieldValueBuffer):
    def __init__(
        self, 
        field_descriptor: FieldDescriptor,
        history_buffer: HistoryBuffer,
        ts: TimeStep
    ):
        super().__init__(field_descriptor)
        self._ts = ts
        self._history_buffer = history_buffer

    def set(self, value: np.ndarray):
        self._history_buffer.store(self._ts, value)

    def get(self) -> np.ndarray:
        return self._history_buffer.at(self._ts)