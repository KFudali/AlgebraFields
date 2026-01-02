from space.base import FieldDescriptor, Space
from space.time import TimeSeries
from .transient_field import TransientField

class VectorTransientField(TransientField):
    def __init__(
        self, space: Space, time_series: TimeSeries
    ):
        super().__init__(
            FieldDescriptor(space, components=space.dim), 
            time_series
        )