from .time_series import TimeSeries
from .time_window import TimeWindow

class TimeWindowManager():
    def __init__(self, time_series: TimeSeries):
        self._time = time_series
        self._current_ts = self._time.step(0)
        self._time_windows = set[TimeWindow]()

    def window(self, offset: int = 0) -> TimeWindow:
        ts = self._time.offset_step(self._current_ts, offset)
        tw = TimeWindow(ts)
        return tw

    def offset_window(self, window: TimeWindow, offset: int) -> TimeWindow:
        ts = self._time.offset_step(window.ts, offset)
        window.assign_ts(ts)

    def step_forward(self, ts_count: int = 1):
        if ts_count < 0: raise RuntimeError()
        for window in self._time_windows:
            self.offset_window(window, ts_count)
        ts = self._time.offset_step(self._current_ts, ts_count)
        self._current_ts = ts
    
    def reset(self):
        self._current_ts = self._time.step(0)
        for window in self._time_windows:
            window.assign_ts(self._current_ts)