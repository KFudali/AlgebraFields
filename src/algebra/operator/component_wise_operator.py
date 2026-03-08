import numpy as np
from .operator_wrapper import OperatorWrapper, Operator
from algebra.exceptions import ShapeMismatchException

class ComponentWiseOperator(OperatorWrapper):
    def __init__(
        self, 
        op: Operator,
        components: int,
    ):
        if op.input_shape != op.output_shape:
            raise ShapeMismatchException((
                "FieldShapedOperator can only be created from operator that has ",
                "matching input and output sizes.\n"
                f"Got sizes: input: {op.input_shape}, output: {op.output_shape}"
            ))
        shape = (components, *op.input_shape)
        self._op = op
        self._components = components
        super().__init__(self._op, shape, shape)

    @property
    def components(self) -> int:
        return self._components

    def _apply(self, field: np.ndarray, out: np.ndarray):
        for comp in self.input_shape[0]:
            self._op.apply(field[comp, :], out[comp, :])