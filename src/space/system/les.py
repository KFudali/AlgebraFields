import numpy as np

from .system import EqSystem
from space.core import FieldOperator, FieldExpression, CallableFieldExpression
from tools.algebra.operator import CallableOperator



from tools.algebra.system import LES
from discr.core.bcs import DiscreteBC

class LESExpr(EqSystem):
    def __init__(self, Ax: FieldOperator, rhs: FieldExpression):
        super().__init__()
        self._Ax = Ax
        self._rhs = rhs
        self._bcs = list[DiscreteBC]()

    def add_bc(self, bc: DiscreteBC):
        if bc in self._bcs: return
        self._bcs.append(bc)

    def _assemble(self) -> LES:
        disc = self._rhs.space.discretization
        def flat_op(field: np.ndarray, out: np.ndarray):
            reshaped_field = disc.reshape(field)
            reshaped_out = disc.reshape(out)
            self._Ax.apply(reshaped_field, reshaped_out)
            out = disc.flatten(reshaped_out)
        flat_shape = (np.prod(self._Ax.input_shape),)
        flat_Ax = CallableOperator(flat_shape, flat_shape, flat_op)
        system = LES(flat_Ax, disc.flatten(self._rhs.eval()))
        for bc in self._bcs:
            bc.apply_to_system(system)
        return system
        
    def solve(self) -> FieldExpression:
        system = self._assemble()
        def solve_reshape() -> np.ndarray:
            value = system.solve(method='cg')
            return self._Ax.reshape(value)
        return CallableFieldExpression(
            self._rhs.space, self._rhs.components, solve_reshape 
        )