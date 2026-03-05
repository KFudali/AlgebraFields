from ..field import Field
from .field_operator_expression import FieldOperatorExpression

from space.time import TimeSeries, explicit


def euler(field: Field, time: TimeSeries) -> FieldOperatorExpression:
    return explicit.EulerTimeDer(field, time)