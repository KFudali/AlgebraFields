from abc import ABC, abstractmethod
import numpy as np

from .shapebound import ShapeBound

class Operator(ShapeBound, ABC): 
    @abstractmethod 
    def _apply(self, array: np.ndarray, out: np.ndarray): pass