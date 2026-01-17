import numpy as np
from .fieldshaped import Space
from space.core import FieldExpression, FieldOperator, ShapeMismatch

class FieldLinearOperator(FieldOperator):
    def __init__(
        self, space: Space, components: int, Ax: FieldOperator, b: FieldExpression
    ):
        super().__init__(space, components)
        if not Ax.input_shape == Ax.output_shape == b.output_shape :
            raise ShapeMismatch(
                "Linear operator components should have matching shapes",
                f"operator input shape: {Ax.input_shape}",
                f"operator input shape: {Ax.output_shape}",
                f"expression output shape: {b.output_shape}",
            )

        self._Ax = Ax
        self._b = b

    def Ax(self) -> FieldOperator: return self._Ax
    
    def b(self) -> FieldExpression: return self._b

    def _apply(self, field: np.ndarray, out: np.ndarray):
        self._Ax.apply(field, out)
        out += self._b.eval()