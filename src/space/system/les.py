from .system import EqSystem
from tools.algebra.expr import Expression, CallableExpression
from tools.algebra.system import LES
from tools.algebra.operator import Operator
from discr.core.bcs import DiscreteBC

class LES(EqSystem):
    def __init__(self, Ax: Operator, rhs: Expression):
        super().__init__()
        self._Ax = Ax
        self._rhs = rhs
        self._bcs = list[DiscreteBC]()

    def add_bc(self, bc: DiscreteBC):
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
            self._rhs.output_shape, system.solve()
        )