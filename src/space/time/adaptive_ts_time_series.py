from .time_series import TimeSeries
from .time_step import TimeStep, TimeStepId

class AdaptiveTimeSeries(TimeSeries):
    def __init__(self, start_time: float, end_time: float):
        assert end_time > start_time
        self._start = start_time
        self._end = end_time
        self._dt = dt
        self._ts_list = list[TimeStep]()
        start, end = self.span
        for i, t in enumerate(range(start, end, self.dt)):
            id = TimeStepId(i)
            ts = TimeStep(id, self._dt)
            self._ts_list.append(ts)
        
    @property 
    def dt(self) -> float: return self._dt
    
    @property 
    def n_steps(self) -> int: return len(self._ts_list)
    
    @property 
    def span(self) -> tuple[float, float]: 
        return tuple(self._start, self._end)

    def step(self, step_n: int) -> TimeStep:
        return self._ts_list[step_n]

    def offset_step(self, step: TimeStep, offset: int) -> TimeStep:
        return self._ts_list[self._ts_list.index(step) + offset]