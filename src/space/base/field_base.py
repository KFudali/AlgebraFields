from abc import ABC, abstractmethod
from space.base import FieldObject, FieldDescriptor
from model.discretization import Discretization
import numpy as np

class AbstractField(FieldObject, ABC):
    def __init__(
        self, field_descriptor: FieldDescriptor
    ):
        super().__init__(field_descriptor)

    @property
    @abstractmethod
    def disc(self) -> Discretization:
        return self.space.disc

    @abstractmethod
    def raw_value(self) -> np.ndarray: pass

    @abstractmethod
    def set_raw_value(self, raw_value: np.ndarray) -> bool: pass