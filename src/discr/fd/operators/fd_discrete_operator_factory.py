from tools.geometry import StructuredGridND
import algebra

from discr.core import DiscreteOperatorsFactory
from discr.fd.domain import FDDomain

from .fd_stencil_operator import FDStencilOperator

class FDDiscreteOperatorsFactory(DiscreteOperatorsFactory):
    def __init__(self, domain: FDDomain):
        self._domain = domain

    @property
    def grid(self) -> StructuredGridND:
        return self._domain.grid

    def eye(self) -> FDStencilOperator:
        contribs = {ax: {} for ax in range(self.grid.ndim)}
        contribs[1] = {0: 1.0}
        stencil = algebra.stencil.Stencil(contribs)
        return FDStencilOperator(self._domain, stencil)

    def laplace(self) -> FDStencilOperator:
        contribs = {}
        for ax in range(self.grid.ndim):
            h = self.grid.ax_spacing(ax)
            left = 1.0 / h**2
            right = 1.0 / h**2
            central = -2.0 / h**2
            second_central = {-1: left, 0: central, 1: right}
            contribs[ax] = second_central
        stencil = algebra.stencil.Stencil(contribs)
        return FDStencilOperator(self._domain, stencil)