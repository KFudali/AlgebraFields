from dataclasses import dataclass
from utils.ids import IntId

class TimeStepId(IntId): pass
class TimeStep():
    def __init__(
        self, id: TimeStepId, dt: float
    ):
        self._id = id
        self._dt = dt
    
    @property
    def id(self) -> TimeStepId: return self._id
    
    @property
    def dt(self) -> float: return self._dt

    def __eq__(self, value):
        if isinstance(value, 'TimeStep'):
            return value.id == self.id
        return False