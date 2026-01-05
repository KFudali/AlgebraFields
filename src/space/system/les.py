from .system import EqSystem
from algebra import Operator, Expression
from algebra.expression import CallableExpression
from algebra.system import LES, BoundaryCondition

class LESExpression(EqSystem):
    def __init__(self, Ax: Operator, rhs: Expression):
        super().__init__()
        self._Ax = Operator
        self._rhs = Expression
        self._bcs = list[BoundaryCondition]()

    def add_bc(self, bc: BoundaryCondition):
        if bc in self._bcs: return
        self._bcs.append(bc)

    def _assemble(self) -> LES:
        rhs = self._rhs.eval()
        system = LES(self._Ax, rhs)
        for bc in self._bcs:
            bc.apply_to_system(system)
        return system
        
    def solve(self) -> Expression:
        system = self._assemble()
        return CallableExpression(
            self._rhs.output_shape, system.solve
        )