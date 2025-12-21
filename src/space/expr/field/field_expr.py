from ..expr import Expr
from space.base import FieldObject

class ShapeMismatchError(Exception): pass
class FieldExpr(FieldObject, Expr): pass