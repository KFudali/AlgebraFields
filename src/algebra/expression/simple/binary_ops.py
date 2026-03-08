from algebra.exceptions import ShapeMismatchException
from .simple_expr import SimpleExpression, Expression

class ExprBinaryOp(SimpleExpression):
    def __init__(
        self, 
        left: Expression, right: Expression, 
        output_shape: tuple[int, ...]
    ):
        self.left = left
        self.right = right
        super().__init__(output_shape)

class ShapeMatchedExprBinaryOp(ExprBinaryOp):
    def __init__(self, left: Expression, right: Expression):
        if left.output_shape != right.output_shape:
            raise ShapeMismatchException(
                (
                    "Can only compose operator from exprs of equal shape.",
                    f"left shape: {left.output_shape}, ",
                    f"right shape: {right.output_shape}"
                )
            )
        super().__init__(left, right, right.output_shape)

class AddExpr(ShapeMatchedExprBinaryOp):
    def __init__(self, left: Expression, right: Expression):
        super().__init__(left, right)

    def eval(self):
        return self.left.eval() + self.right.eval()

class SubtractExpr(ShapeMatchedExprBinaryOp):
    def __init__(self, left: Expression, right: Expression):
        super().__init__(left, right)

    def eval(self):
        return self.left.eval() - self.right.eval()

class ElementWiseMulExpr(ShapeMatchedExprBinaryOp):
    def __init__(self, left: Expression, right: Expression):
        super().__init__(left, right)

    def eval(self):
        return self.left.eval() * self.right.eval()

class ElementWiseDivExpr(ShapeMatchedExprBinaryOp):
    def __init__(self, left: Expression, right: Expression):
        super().__init__(left, right)

    def eval(self):
        return self.left.eval() / self.right.eval()