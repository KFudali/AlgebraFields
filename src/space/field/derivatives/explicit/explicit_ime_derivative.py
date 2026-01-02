import numpy as np
from space.field.transient import TransientField
from space.field.base import Field
from space.time import TimeStep

class ExplicitTimeDerivative(TransientField):
    def __init__(
        self,
        transient_field: TransientField
    ):
        self._source_field = transient_field

    def at(self, ts: TimeStep) -> Field: pass

    def _value(self, ts: TimeStep) -> FieldValue:
        prev_ts = self._time.offset_step(ts, -1)
        ts_value = self._source_field.at(ts).value()
        prev_ts_value = self._source_field.at(prev_ts).value()
        return (ts_value - prev_ts_value) / ts.dt

FieldValue() -> basic object containing setters and getters to raw values
FieldValueBuffer()
    get()->FieldValue()
    set()->FieldUpdate()

EqSpace()
    time_series()

Field(SpaceBound)

ScalarField(SpaceBound)
VectorField(SpaceBound)
TransientScalarField(ScalarField)
    at() -> ScalarField
TransientVectorField(VectorField)
    at() -> VectorField
    component(int) -> Field
VectorExplicitTimeDerivative(VectorTransientField)
ScalarExplicitTimeDerivative(ScalarTransientField)


Field
    value() -> FieldValue
    update() -> FieldUpdate
    
    at(tw: TimeWindow) -> Field

    space() -> space 
    components() -> int

LinearOperator()
    __init__(field)
    eval() -> FieldValue()
    shape
    stencils().A
    stencils().b
    
    apply(x, out)

Algorithm composition -> only Fields, FieldValues, FieldUpdates etc

A + B -> FieldValue
C.update(A+B)
LinearOperator()
    eval -> FieldValue


FieldShape(space, components)

Field()
    def __init__(space, components: int):
        self._value_buffer = ValueBuffer(space.shape, components)

    value() -> FieldValue():
        return FieldValue(self.shape, self._value_buffer.get)

    update(value) -> FieldUpdate():
        return FieldUpdate(field = self, value)

    at(tw) -> FieldValue
        return FieldValue(self.shape, self._value_buffer.get(tw))

FieldValue()
    def __init__(self, shape: FieldShape, value_getter):
        self._shape = shape
        self._getter = value_getter

    def eval(self) -> np.ndarray:
        return self._getter()

FieldUpdate()
    def __init__(self, field, value):
        self._field = Field
        self._value = value

    def eval():
        self._field._value_buffer.set(self._value.eval())
    
LaplaceOpertator(LinearOperator):
    def __init__(field):
        self._field = field

    def value(self) -> FieldValue:
        def getter():
            value = np.zeros(self._field.shape)
            self._apply(self._field.value.eval(), value)
            return value
        return FieldValue(self._field.shape, getter())

    def _apply(self, x: np.ndarray, out: np.ndarray):
        self._space.discretization.laplace()....

class Operator():
    def value(self) -> FieldValue()
    def _apply(self, x: np.ndarray, out: np.ndarray)
    def _return_apply(self, x) -> np.ndarray

LESSystem()
    def __init__(
        A: Operator,
        rhs: FieldValue
    ):
        self._A = A

    def rhs() ->FieldValue: return self._rhs

    def A() -> MultOperator: return self._A

    def apply_bc(bc: BoundaryCondition):
        # modify self._A and rhs based on condiditon
        bc.linear_apply(A = A, b = rhs)

LESSolve(Expr):
    def __init__(system: LESSystem):
        self._system = system

    def solve(self) -> FieldValue:
        def cg_solve():
            result = np.zeros(shape = self._system.A.shape)
            def returning_operator(x: operator):
                self._system.A._apply(x, result)
                return result
            A = LinearOperator(
                shape = self._system.shape,
                matvec = returning_operator
            )
            x, info = scipy.sparse.linalg.cg(A, self._system.rhs.eval())
        return FieldValue(self._system.shape, cg_solve())

Field()
    __init__(space, components)
        self._value_buffer = HistoryBuffer() #for storing t-1 t-2 if needed larger storage will be external
    shape
    value() -> FieldValue
    update(FieldValue) -> Update

    FieldValue(space, components, some field shape descriptor, get_array)
        eval() -> np.ndarray
            return get_array()

    FieldUpdate(


LinearOperator(space)
    apply(field) -> FieldValue

Laplace(field).stencil
Laplace(field).eval()
Laplace(field).expr() -> F

FieldValue + FieldValue
FieldValue()

class Field():
    def init(desc, value_buffer):
        value_buffer.get(ts) -> FieldValue()
        value_buffer.update(ts) -> FieldUpdate()
    
    value() -> FieldValue
    prev(int) -> Field
    at() -> Field

class SteadyValueBuffer():
    def __init__(self):
        self._value = np.zeros()

    def get(self, ts) -> FieldValue
        def value_getter():
            return self._value
        return FieldValue(self._value.shape, value_getter) 

    def update(self, ts, value) -> FieldValue:
        def value_setter():
            return self._value
        return Field(self._value.shape, value_getter) 

class TransientValueBuffer():