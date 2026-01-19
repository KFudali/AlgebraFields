
from abc import ABC, abstractmethod
from .core.operator import Operator
from ..expr.core.expression import Expression

class LinearOperator(Operator, ABC):
    def __init__(
        self,
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...]
    ):
        super().__init__(input_shape, output_shape)

    @abstractmethod
    def Ax(self) -> Operator: pass
    
    @abstractmethod
    def b(self) -> Expression: pass