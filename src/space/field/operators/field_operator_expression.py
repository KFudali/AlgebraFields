import numpy as np
from typing import Self

from algebra.operator import OperatorExpression, Operator
from ..core import FieldShaped, FieldValue

class FieldOperatorExpression(OperatorExpression, FieldShaped):
    def __init__(self, input: FieldValue, operator: Operator):
        OperatorExpression.__init__(self, input, operator)
        FieldShaped.__init__(self, input.space, input.components)

    def _new(self, input: FieldValue, operator: Operator) -> Self:
        return FieldOperatorExpression(input, operator)

    def eval(self) -> np.ndarray:
        out = np.zeros(shape=(self.components, self.output_shape))
        input = self._input.eval()
        for comp in range(self.components):
            self._operator.apply(input[comp][...], out[comp][...])
        return out