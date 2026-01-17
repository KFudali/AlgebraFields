from tools.algebra.exceptions import ShapeMismatchException
from .core_expression import CoreExpression


class ExprBinaryOp(CoreExpression):
    def __init__(
        self, 
        left: CoreExpression, 
        right: CoreExpression, 
        output_shape: tuple[int, ...]
    ):
        self.left = left
        self.right = right
        super().__init__(output_shape)

class ShapeMatchedExprBinaryOp(ExprBinaryOp):
    def __init__(self, left: CoreExpression, right: CoreExpression):
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
    def __init__(self, left: CoreExpression, right: CoreExpression):
        super().__init__(left, right)

    def eval(self):
        return self.left.eval() + self.right.eval()

class SubtractExpr(ShapeMatchedExprBinaryOp):
    def __init__(self, left: CoreExpression, right: CoreExpression):
        super().__init__(left, right)

    def eval(self):
        return self.left.eval() - self.right.eval()

class ElementWiseMulExpr(ShapeMatchedExprBinaryOp):
    def __init__(self, left: CoreExpression, right: CoreExpression):
        super().__init__(left, right)

    def eval(self):
        return self.left.eval() * self.right.eval()

class ElementWiseDivExpr(ShapeMatchedExprBinaryOp):
    def __init__(self, left: CoreExpression, right: CoreExpression):
        super().__init__(left, right)

    def eval(self):
        return self.left.eval() / self.right.eval()


def matmul_shape(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    if len(a) == 0 or len(b) == 0:
        raise ShapeMismatchException("Scalars not supported in matmul")

    # Vector @ vector
    if len(a) == 1 and len(b) == 1:
        if a[0] != b[0]:
            raise ShapeMismatchException("Vector dot requires same length")
        return ()

    # Vector @ matrix
    if len(a) == 1:
        if a[0] != b[-2]:
            raise ShapeMismatchException("Incompatible shapes for matmul")
        return b[:-2] + (b[-1],)

    # Matrix @ vector
    if len(b) == 1:
        if a[-1] != b[0]:
            raise ShapeMismatchException("Incompatible shapes for matmul")
        return a[:-2] + (a[-2],)

    # General case (..., m, k) @ (..., k, n)
    if a[-1] != b[-2]:
        raise ShapeMismatchException("Incompatible shapes for matmul")

    # Broadcast batch dimensions
    batch_a = a[:-2]
    batch_b = b[:-2]

    if len(batch_a) != len(batch_b):
        raise ShapeMismatchException("Batch dimensions not aligned")

    batch = []
    for da, db in zip(batch_a, batch_b):
        if da == db:
            batch.append(da)
        else:
            raise ShapeMismatchException("Batch dimensions must match exactly")

    return tuple(batch) + (a[-2], b[-1])


class MatMulExpr(ExprBinaryOp):
    def __init__(self, left: CoreExpression, right: CoreExpression):
        matmul_shape(left.output_shape, right.output_shape)
        output_shape = (left.output_shape[0], right.output_shape[1])
        super().__init__(left, right, output_shape)

    def eval(self):
        return self.left.eval() @ self.right.eval()