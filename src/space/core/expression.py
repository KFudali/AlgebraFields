from abc import ABC, abstractmethod
import numpy as np

from .shapebound import ShapeBound

class Expression(ShapeBound, ABC):
    @abstractmethod
    def eval(self) -> np.ndarray:
        pass

