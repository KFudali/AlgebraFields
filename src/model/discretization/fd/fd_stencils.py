# import numpy as np
# from ..discretization import Stencils
# from .domain import FDDomain
# from .fd_operators import FDOperators
# from model.geometry.grid import StructuredGridND

# class FDStencils(Stencils):
#     def __init__(self, domain: FDDomain):
#         self._domain = domain
#         self._ops = FDOperators(domain)
#         super().__init__()

#     @property
#     def grid(self) -> StructuredGridND:
#         return self._domain.grid

#     def laplace(self):
#         shape = self.grid.shape
#         ndim = self.grid.ndim
        
#         N = np.prod(shape)
#         L = np.zeros((N, N))

#         for flat in range(N):
#             idx = np.unravel_index(flat, shape)
#             for axis in range(ndim):
#                 spacing = self.grid.ax_spacing(axis)
#                 h2 = spacing**2
#                 L[flat, flat] += -2.0 / h2
#                 for direction in (-1, 1):
#                     nbr = list(idx)
#                     nbr[axis] += direction
#                     if 0 <= nbr[axis] < shape[axis]:
#                         nbr_flat = np.ravel_multi_index(nbr, shape)
#                         L[flat, nbr_flat] = 1.0 / h2
#         return L

#     def grad(self) -> np.ndarray:
#         return super().grad()