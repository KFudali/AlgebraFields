from abc import ABC, abstractmethod
import numpy as np

from spacer.base import AbstractField
from .operator import Operator
from model.discretization import Discretization

class LinearOperator(Operator, ABC):
    def __init__(self, field: AbstractField):
        super().__init__(field)

    @property
    @abstractmethod
    def stencil(self, discretization: Discretization) -> np.ndarray: pass