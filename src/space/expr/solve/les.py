import numpy as np
import functools

from space.base.bc import BC
from ..operators import LinearOperator
from ..field import FieldValue, FieldExpr


class LESSolve(FieldExpr):
    def __init__(
        self,
        linear_operator: LinearOperator,
        rhs: FieldValue
    ):
        super().__init__(linear_operator.desc)
        assert linear_operator.same_shape(rhs)
        self._linear_operator = linear_operator
        self._rhs = rhs
        self._bcs = set[BC]()

    def apply_bc(self, bc: BC):
        self._bcs.add(bc)

    def assemble(self):
        A = self._linear_operator.stencil
        b = self._rhs.eval()
        for bc in self._bcs:
            bc.apply_linear(A, b)
        return A, b

    def solve(self) -> FieldValue:
        A, b = self.assemble()
        value_callback = functools.partial(np.linalg.solve, A, b)
        return FieldValue(self.desc, value_callback)