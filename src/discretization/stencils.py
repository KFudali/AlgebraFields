from abc import ABC, abstractmethod
import numpy as np

class Stencils(ABC):
    @abstractmethod
    def laplace(self) -> np.ndarray: pass

    @abstractmethod
    def grad(self) -> np.ndarray: pass