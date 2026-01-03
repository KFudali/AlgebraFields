from abc import ABC, abstractmethod
import numpy as np
from ...time.time_step import TimeStep
from ...time.time_series import TimeSeries

class HistoryBuffer(ABC):
    def __init__(
        self, 
        time_series: TimeSeries,
        buffer_size: int = 10
    ):
        self._time = time_series
        self._buffer_size = buffer_size

    @abstractmethod
    def at(self, ts: TimeStep) -> np.ndarray: pass

    @abstractmethod
    def store(self,  ts: TimeStep, value: np.ndarray): pass