import numpy as np

from algebra.expression import SolveLESExpression
from algebra.system import LES
from model.discrete import DiscreteBC

class SolveLES():
    def __init__(self, ):
        super().__init__(les, method, maxiter, tol)
        self._bcs =  list[DiscreteBC]()

    def apply_bc(self, bc: DiscreteBC):
        if bc in self._bcs: return
        self._bcs.append(bc)
    
    def _assemble(self):
        for bc in self._bcs:
            bc.apply_to_system(self._system)

    def eval(self) -> np.ndarray:
        self._assemble()
        return super().eval()