from spacer.base import FieldObject
from .field_value_buffer import FieldValueBuffer
from model.discretization import Discretization

class AbstractField(FieldObject):
    def __init__(
        self, 
        value_buffer: FieldValueBuffer
    ):
        super().__init__(value_buffer.desc)
        self._value = value_buffer

    @property
    def disc(self) -> Discretization:
        return self.space.disc
    
    @property
    def raw_value(self) -> FieldValueBuffer:
        return self._value