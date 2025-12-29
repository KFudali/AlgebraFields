from .base import SpaceBound, Space
from .field.steady import ScalarField, VectorField
from .field.transient import ScalarTransientField, VectorTransientField
from .time import TimeSeries

class SteadyFieldFactory(SpaceBound):
    def __init__(self, space: Space):
        super().__init__(space)
    
    def scalar(self) -> ScalarField:
        return ScalarField(self._space)

    def vector(self) -> VectorField:
        return VectorField(self._space)

class TransientFieldFactory(SpaceBound):
    def __init__(self, space: Space):
        super().__init__(space)
    
    def scalar(self, time_series: TimeSeries) -> ScalarTransientField:
        return ScalarTransientField(self._space, time_series)

    def vector(self, time_series: TimeSeries) -> VectorTransientField:
        return VectorTransientField(self._space, time_series)

class FieldFactory(SpaceBound):
    def __init__(self, space: Space):
        super().__init__(space)
        self._steady = SteadyFieldFactory(space)
        self._transient = TransientFieldFactory(space)

    @property
    def steady(self) -> SteadyFieldFactory:
        return self._steady
    
    @property
    def transient(self) -> SteadyFieldFactory:
        return self._transient