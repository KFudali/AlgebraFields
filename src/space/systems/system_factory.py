from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..space_bound import SpaceBound, Space
from .les import LES
import expr

class SystemFactory(SpaceBound):
    def __init__(self, space):
        super().__init__(space)

    def les(
        self,
        linear_operator: expr.operators.LinearOperator,
        rhs: expr.FieldValue,
    ):
        return LES(self.space, linear_operator, rhs)