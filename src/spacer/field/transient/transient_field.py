from ..base import Field
from spacer.base import FieldDescriptor, AbstractField
from spacer.time import TimeSeries, TimeStep

from .history_buffer import HistoryBuffer
from .transient_value_buffer import TransientValueBuffer

class TransientField(AbstractField):
    def __init__(
        self, 
        field_descriptor: FieldDescriptor,
        time_series: TimeSeries
    ):
        super().__init__(field_descriptor)
        self._time = time_series
        self._history = HistoryBuffer(self._time)

    def at(self, ts: TimeStep) -> Field:
        if ts in self._time:
            value = TransientValueBuffer(self.desc, self._history, ts)
            field = Field(value)
            return field