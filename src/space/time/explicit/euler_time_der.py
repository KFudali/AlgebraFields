import numpy as np
from space.core import FieldLinearOperator, CallableFieldOperator
from space.field import Field
from ..field_time_der import FieldTimeDerivative


class EulerTimeDerivative(FieldTimeDerivative):
    def __init__(self, field: Field):
        super().__init__(field, required_time_steps=1)

    def _calculate(self) -> np.ndarray:
        return (self.field.value().eval() - self.field.prev_value(1).eval()) / self._dt()
    
    def op(self) -> FieldLinearOperator:
        def ones_op(field: np.ndarray, out: np.ndarray):
            out[:] += field[:] / self._dt()
        op = CallableFieldOperator(self.space, self.components, ones_op)
        exp = -self.field.prev_value(1) / self._dt()
        return FieldLinearOperator(self.space, self.components, op, exp)