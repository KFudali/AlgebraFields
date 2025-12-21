from abc import ABC, abstractmethod
import numpy as np

from .operator import Operator
from space.field import Field
from model.discretization import Discretization

class LinearOperator(Operator, ABC):
    def __init__(self, field: Field):
        super().__init__(field)

    @property
    @abstractmethod
    def stencil(self, discretization: Discretization) -> np.ndarray: pass