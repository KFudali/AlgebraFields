from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..space_bound import SpaceBound, Space

from ..boundary import BC
import expr

class LES(SpaceBound):
    def __init__(
        self,
        space: Space,
        linear_operator: expr.operators.LinearOperator,
        rhs: expr.FieldValue
    ):
        super().__init__(space)
        self._linear_operator = linear_operator
        self._rhs = rhs
        self._bcs = set[BC]()
        self.A = None
        self.b = None

    def apply_bc(self, bc: BC):
        self._bcs.add(bc)

    def assemble(self):
        self.A = self._linear_operator.stencil
        self.b = self._rhs.eval()
        for bc in self._bcs:
            bc.apply_linear(self.A, self.b)
        