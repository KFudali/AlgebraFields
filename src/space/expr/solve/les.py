from typing import Callable
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
        self._linear_operator = linear_operator
        self._rhs = rhs
        self._bcs = set[BC]()

    def apply_bc(self, bc: BC):
        self._bcs.add(bc)

    def assemble(self):
        A = self._linear_operator.stencil
        b = self._rhs.eval().ravel()
        for bc in self._bcs:
            bc.apply_linear(A, b)
        return A, b

    def solve(self) -> FieldValue:
        A, b = self.assemble()
        value_callback = functools.partial(np.linalg.solve, A, b)
        def solution_callback():
            value = value_callback()
            return value.reshape(self._rhs.shape)
        return FieldValue(self.desc, solution_callback)
    
    def eval(self) -> np.ndarray:
        A, b = self.assemble()
        return np.linalg.solve(A, b).reshape(self._rhs.shape)