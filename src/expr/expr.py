from abc import abstractmethod, ABC

from ..space.field_bound import FieldBound
from ..space.field.field_value import FieldValue

class Expr(FieldBound, ABC):
    @abstractmethod
    def eval(self) -> FieldValue: pass