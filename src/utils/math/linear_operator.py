from dataclasses import dataclass
import numpy as np

@dataclass
class LinOp:
    A: np.ndarray
    b: np.ndarray