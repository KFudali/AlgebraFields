import numpy as np
from .field_expression import FieldExpression
from tools.algebra.expr import Expression
from tools.algebra.exceptions import ShapeMismatchException
from space.core import Space

class FieldExpressionWrapper(FieldExpression):
    def __init__(
        self, space: Space, components: int, exp: Expression
    ):
        super().__init__(space, components)
        if exp.output_shape != self.shape:
            raise ShapeMismatchException(
                "Can only wrap expression that mathces field shape",
                f"Field shape: {self.shape}",
                f"Expression output_shape: {exp.output_shape}"
            )
        self._exp = exp

    def eval(self) -> np.ndarray:
        return self._exp.eval()

# class FieldExprUnaryOp(FieldExpression, unary_ops.ExprUnaryOp):
#     def __init__(self, operand: FieldExpression):
#         FieldExpression.__init__(
#             self,
#             space=operand.space,
#             components=operand.components,
#         )
#         unary_ops.ExprUnaryOp.__init__(self, operand, operand.shape)

# class FieldExprBinaryOp(FieldExpression, binary_ops.ExprBinaryOp):
#     def __init__(self, left: FieldExpression, right: FieldExpression):
#         if left.space != right.space:
#             raise ValueError("Field spaces must match")
#         if left.components != right.components:
#             raise ValueError("Field components must match")

#         FieldExpression.__init__(
#             self,
#             space=left.space,
#             components=left.components,
#         )
#         binary_ops.ExprBinaryOp.__init__(self, left, right, left.shape)
