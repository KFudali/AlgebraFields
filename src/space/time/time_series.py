from abc import ABC, abstractmethod
from typing import Optional
from .time_step import TimeStep

class TimeSeries(ABC):
    @property 
    @abstractmethod
    def n_steps(self) -> int: pass
    
    @property 
    @abstractmethod
    def span(self) -> tuple[float, float]: pass

    @abstractmethod
    def step(self, step_n: int) -> Optional[TimeStep]: pass

    @abstractmethod
    def offset_step(self, step: TimeStep, offset: int) -> TimeStep: pass