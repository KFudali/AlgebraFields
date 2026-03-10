
from space.field.field import Field
from space.field.operators import FieldOperatorExpression
import algebra

class EulerTimeDer(FieldOperatorExpression):
    def __init__(self, field: Field):

        field.save_past(2)
        
        self._discr = field.space.discretization
        self._time = field.space.time

        op = self._discr.operators.eye() / self._dt()
        comp_op = algebra.operator.ComponentWiseOperator(op, field.components)
        rhs = -field.past(1).value() / self._dt()
        op = algebra.operator.CombinedOperator(comp_op, rhs)
        super().__init__(field.value(), op)

    def _dt(self) -> algebra.ScalarExpression:
       return algebra.ScalarExpression(self._time.dt)