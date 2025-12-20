from .expr import Expr
from ..space.field_bound import FieldBound

class ShapeMismatchError(Exception): pass
class FieldExpr(FieldBound, Expr): pass