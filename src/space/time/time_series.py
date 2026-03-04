class TimeSeries():
    def __init__(self):
        self._dts = []

    def advance(self, dt: float):
        self._dts.append(dt)

    def last_dt(self) -> float:
        return self._dts[-1]
    
    def time(self) -> float:
        return sum(self._dts, 0.0)
