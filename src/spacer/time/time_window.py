from .time_step import TimeStep

class TimeWindow():
    def __init__(
        self, ts: TimeStep, 
    ):
        self._ts = ts

    def assign_ts(self, ts: TimeStep):
        self._ts = ts

    @property
    def ts(self) -> TimeStep:
        return self._ts