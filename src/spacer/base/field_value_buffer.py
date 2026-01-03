from abc import ABC, abstractmethod
from . import FieldObject, FieldDescriptor
import numpy as np

class FieldValueBuffer(FieldObject, ABC):
    @abstractmethod
    def get(self) -> np.ndarray: pass
    @abstractmethod
    def set(self, value: np.ndarray): pass
