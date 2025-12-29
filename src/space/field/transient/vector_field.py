from space.base import FieldDescriptor,  Space
from .transient import TransientField
from ..time import TimeSeries

class VectorTransientField(TransientField):
    def __init__(
        self, space: Space, time_series: TimeSeries
    ):
        super().__init__(
            FieldDescriptor(space, components=space.dim), 
            time_series
        )