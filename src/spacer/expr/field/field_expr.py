from ..expr import Expr
from spacer.base import FieldObject

class ShapeMismatchError(Exception): pass
class FieldExpr(FieldObject, Expr): pass