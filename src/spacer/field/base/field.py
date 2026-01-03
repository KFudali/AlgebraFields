import numpy as np
from spacer.base import  AbstractField, bc, FieldValueBuffer
from model.discretization import Discretization

from spacer.expr.field import FieldValue
from .field_operator_factory import FieldOperatorFactory

class FieldShape(): pass
class FieldShapeBound():
    @property
    def shape(self): pass
    
    @property
    def components(self) -> int: pass

class Expression(FieldShapeBound):
    def _eval(self) -> np.ndarray: pass

class AbstractField(FieldShapeBound):
    def _get(self, window: TimeWindow) -> np.ndarray: pass
    def _set(self, window: TimeWindow, value: np.ndarray): pass

class FieldValue(Expression):
    def __init__(self, field: AbstractField, window = None):
        self._field = field
        self._window = window

    def _eval(self) -> np.ndarray:
        self._field._get(self._window)

class FieldUpdate():
    def __init__(
        self, 
        field: AbstractField, 
        expression: Expression,
        window = None
    ):
        self._field = field
        self._expr = expression
        self._window = window

    def _eval(self):
        self._field._set(self._window, self._expr._eval())

class Field(AbstractField):
    def __init__(self, shape: FieldShape):
        FieldShapeBound.__init__(shape)

    def at(self, window: TimeWindow) -> FieldValue:
        return FieldValue(self, self._value_buffer.get(window))

    def value(self) -> FieldValue:
        return self.at(eq_space.window())

    def update(self, expression: Expression) -> FieldUpdate:
        return FieldUpdate(self, expression)



class LaplaceOperator(Operator, FieldShapeBound):
    def __init__(self, shape: FieldShape):
        FieldShapeBound.__init__(shape)
    
    def _apply(self, array: np.ndarray, out: np.ndarray):
        self.space.discretization.laplace... (
            FieldShapeBound has access to space 
            and therefore to disc
        )

class AppliedOperator(Expression):
    def __init__(self, op: Operator, expr: Expression):
        super().__init__()
        self._op = op
        self._expr = expr

    def _eval(self) -> np.ndarray:
        x = self._expr.eval()
        out = np.zeros_like(x)
        self._op.apply(x, out)
        return out

class LESSystem():
    def __init__(
        self, lhs: Operator, rhs: Expression
    ):
        self._lhs = lhs 
        self._rhs = rhs 
        self._bcs = set[BoundaryCondition]()

    def apply_bc(self, bc: BoundaryCondition):
        self._bcs.add(bc)

    def _assemble(self):
        rhs = self._rhs._eval()
        for bc in self._bcs:
            self._lhs.modify(bc.apply_linear(self._lhs, rhs))
            # This could internally chain operators to add 
            # different evaluation at boundary id, is this good?
        self._rhs_value = rhs

    def solve(self) -> Expression:
        if self._rhs_value is None:
            self._assemble()
            def cg_solve():
                def operator(x: np.ndarray):
                    out = np.zeros_like(x)
                    self._lhs._apply(x, out)
                    return out
                x, info = cg(operator, self._rhs_value)
            return Expression(self.shape, cg_solve)

class ExplicitEulerTimeDerivative(AbstractField):
    def __init__(self, field: Field):
        super().__init__(field.shape)
        field.save_past(1)
        self._source_field = field

    # Shouldt really have update, or should it? maybe it could

    def _get(self, tw: Window) -> np.ndarray:
        diff = self._source_field._get(tw) - self._source_field._get(tw-1)
        return diff / tw.dt