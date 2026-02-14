import numpy as np
import numbers


from discr.fd import FDDiscretization
from .fd_stencil import FDStencil
from discr.core.domain import BoundaryId

class FDStencilOperator:
    def __init__(self, disc: FDDiscretization):
        self._disc = disc
        self._stencils: dict[BoundaryId, FDStencil] = {}
        self._interior_stencil: FDStencil = None

    def set_interior_stencil(self, stencil: FDStencil):
        self._interior_stencil = stencil

    def set_boundary_stencil(self, boundary_id: BoundaryId, stencil: FDStencil):
        self._stencils[boundary_id] = stencil

    def apply(self, field: np.ndarray, out: np.ndarray):
        for bc_id, stencil in self._stencils.items():
            boundary = self._disc.domain.boundary(bc_id)
            bc_region = stencil.boundary_region(
                field.shape, boundary.axis, boundary.inward_dir
            )
            stencil.apply_to_region(field, out, bc_region)
        self._interior_stencil.apply(field, out)

    def copy(self) -> "FDStencilOperator":
        new = FDStencilOperator(self._disc)
        new._interior_stencil = (
            self._interior_stencil.copy()
            if self._interior_stencil is not None
            else None
        )
        new._stencils = {
            bid: stencil.copy()
            for bid, stencil in self._stencils.items()
        }
        return new

    def _combine(self, other: "FDStencilOperator", op):
        if not isinstance(other, FDStencilOperator):
            return NotImplemented

        if self._disc is not other._disc:
            raise ValueError("Cannot combine operators from different discretizations.")

        result = FDStencilOperator(self._disc)

        if self._interior_stencil and other._interior_stencil:
            result._interior_stencil = op(
                self._interior_stencil,
                other._interior_stencil
            )
        elif self._interior_stencil:
            result._interior_stencil = self._interior_stencil.copy()
        elif other._interior_stencil:
            result._interior_stencil = other._interior_stencil.copy()

        all_bids = set(self._stencils) | set(other._stencils)

        for bid in all_bids:
            s1 = self._stencils.get(bid)
            s2 = other._stencils.get(bid)

            if s1 and s2:
                result._stencils[bid] = op(s1, s2)
            elif s1:
                result._stencils[bid] = s1.copy()
            elif s2:
                result._stencils[bid] = s2.copy()

        return result

    def __add__(self, other):
        return self._combine(other, lambda a, b: a + b)

    def __iadd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._combine(other, lambda a, b: a - b)

    def __isub__(self, other):
        return self.__sub__(other)

    def __neg__(self):
        result = self.copy()

        if result._interior_stencil:
            result._interior_stencil = -result._interior_stencil

        for bid in result._stencils:
            result._stencils[bid] = -result._stencils[bid]

        return result

    def _scale(self, scalar: float):
        result = self.copy()

        if result._interior_stencil:
            result._interior_stencil = result._interior_stencil * scalar

        for bid in result._stencils:
            result._stencils[bid] = result._stencils[bid] * scalar

        return result

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            return self._scale(other)

        if isinstance(other, FDStencilOperator):
            return self._combine(other, lambda a, b: a * b)

        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, scalar):
        if not isinstance(scalar, numbers.Number):
            return NotImplemented
        return self._scale(1.0 / scalar)

    def __itruediv__(self, scalar):
        return self.__truediv__(scalar)