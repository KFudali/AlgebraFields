from abc import ABC, abstractmethod
import numpy as np
from .fieldshaped import FieldShaped

class AbstractField(FieldShaped, ABC):
    @abstractmethod
    def _set_current(self, value: np.ndarray): pass

    @abstractmethod
    def _get_current(self) -> np.ndarray: pass

    @abstractmethod
    def _get_past(self, past_step: int = 1) -> np.ndarray: pass

    @abstractmethod
    def _advance(self): pass