
from space.field.field import Field
from space.field.operators import FieldOperatorExpression
from ..time_series import TimeSeries
import algebra

class EulerTimeDer(FieldOperatorExpression):
    def __init__(self, field: Field, time: TimeSeries):
        field.save_past(1)
        
        self._discr = field.space.discretization
        self._time = time

        op = self._discr.operators.eye() / self._dt()
        self._rhs = field.past(1).value() / self._dt()
        super().__init__(field.value(), op)

    def rhs(self) -> algebra.Expression:
        return self._rhs

    def _dt(self) -> algebra.ScalarExpression:
       return algebra.ScalarExpression(self._time.last_dt)