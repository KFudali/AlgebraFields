from ..field import Field
from .field_operator_expression import FieldOperatorExpression


def laplace(field: Field) -> FieldOperatorExpression:
    discr = field.space.discretization
    laplace = discr.operators.laplace()
    return FieldOperatorExpression(field.value(), laplace)