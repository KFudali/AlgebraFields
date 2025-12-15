from dataclasses import dataclass
import numpy as np

@dataclass
class LinSys:
    A: np.ndarray
    b: np.ndarray