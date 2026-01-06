from tools.algebra import Operator, Expression
from ..fieldshaped import FieldShaped, Space

class FieldOperator(Operator, FieldShaped):
    def __init__(self, space: Space, components: int):
        FieldShaped.__init__(self, space, components)
        Operator.__init__(self, self.shape, self.shape)

class FieldExpression(Expression, FieldShaped):
    def __init__(self, space: Space, components: int):
        FieldShaped.__init__(self, space, components)
        Expression.__init__(self, self.shape)

class FieldBC(): 
    def __init__(self, target_shape: FieldShaped):
        pass

class LESExpr():
    def __init__(self, Ax: FieldOperator, rhs: FieldExpression):
        self._Ax = Ax
        self._rhs = rhs

    def add_bc(self, bc: FieldBC):
        if bc in self._bcs: return
        self._bcs.append(bc)

    def _assemble(self) -> LES:
        rhs = self._rhs.flat.eval()
        Ax = self._Ax.flat
        system = LES(Ax, rhs) #requires flat rhs and Ax as operator N by N
        for bc in self._bcs:
            bc.flat.apply_to_system(system)
        return system
        
    def solve(self) -> FieldExpression:
        system = self._assemble()
        def solve_reshape():
            result = system.solve()
            result.reshape(self._rhs.shape)
            return result
        return FieldExpression(self._rhs.shape, solve_reshape)