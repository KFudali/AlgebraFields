import numpy as np
from typing import Callable

from space.core.shapebound import FieldShape
from space.core.expression import Expression


class GetterExpression(Expression):
    def __init__(self, shape: FieldShape, getter: Callable[[], np.ndarray]):
        super().__init__(shape)
        self._getter = getter

    def eval(self) -> np.ndarray:
        value = self._getter()
        msg = (
            "Value getter used for creating this expression does not match field shape."
        )
        self.assert_array_shape(value.shape, msg)
        return value
