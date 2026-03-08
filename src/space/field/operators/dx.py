from ..field import Field
from .field_operator_expression import FieldOperatorExpression
import algebra

def laplace(field: Field) -> FieldOperatorExpression:
    laplace = field.space.discretization.operators.laplace()
    op = algebra.operator.CombinedOperator(
        laplace, algebra.expression.ZeroExpression(laplace.output_shape)
    )
    field_shaped = algebra.operator.ComponentWiseOperator(op, field.components)
    return FieldOperatorExpression(field.value(), field_shaped)