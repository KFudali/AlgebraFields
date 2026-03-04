from __future__ import annotations

from typing import Self
import numbers
import numpy as np

import algebra
from algebra.exceptions import ShapeMismatchException
import tools.region as region
from algebra.stencil import Stencil
from discretization.fd.domain import FDDomain
from discretization.core.domain import BoundaryId


class FDStencilOperator(algebra.Operator):
    def __init__(self, domain: FDDomain, stencil: Stencil):
        super().__init__(domain.grid.shape, domain.grid.shape)
        self._domain = domain
        self._interior_stencil = stencil

        self._boundary_stencils: dict[BoundaryId, Stencil] = {}
        for bid in domain.boundaries:
            self._boundary_stencils[bid] = stencil

    @property
    def domain(self) -> FDDomain:
        return self._domain

    @property
    def interior_stencil(self) -> Stencil:
        return self._interior_stencil

    @property
    def boundary_stencils(self) -> dict[BoundaryId, Stencil]:
        return self._boundary_stencils

    @property
    def stencils(self) -> list[Stencil]:
        return list(self._boundary_stencils.values()) + [self._interior_stencil]

    def apply(self, field: np.ndarray, out: np.ndarray):
        interior = region.interior(
            field.shape, self._interior_stencil.ax_ranges(),
        )
        self._interior_stencil.apply_to_region(field, out, interior)

        for bid, stencil in self._boundary_stencils.items():
            boundary = self._domain.boundary(bid)
            stencil.apply_to_region(field, out, boundary.region)

    def _new(
        self,
        *,
        interior: Stencil,
        boundaries: dict[BoundaryId, Stencil],
    ) -> Self:
        op = self.__class__(self._domain, interior)
        op._boundary_stencils = boundaries
        return op

    def __neg__(self) -> Self:
        return self._new(
            interior=-self._interior_stencil,
            boundaries={bid: -st for bid, st in self._boundary_stencils.items()}
        )

    def __add__(self, other: FDStencilOperator) -> Self:
        if not isinstance(other, FDStencilOperator):
            return NotImplemented
        if (
            self.input_shape != other.input_shape
            or self.output_shape != other.output_shape
        ):
            raise ShapeMismatchException("Operator shape mismatch")

        return self._new(
            interior=self._interior_stencil + other._interior_stencil,
            boundaries={
                bid: self._boundary_stencils[bid]
                + other._boundary_stencils[bid]
                for bid in self._boundary_stencils
            },
        )

    def __mul__(self, other: float) -> Self:
        if not isinstance(other, numbers.Number):
            return NotImplemented
        other = float(other)
        return self._new(
            interior=self._interior_stencil * other,
            boundaries={bid: st * other for bid, st in self._boundary_stencils.items()}
        )

    def __truediv__(self, other: float) -> Self:
        if not isinstance(other, numbers.Number):
            return NotImplemented
        other = float(other)
        return self._new(
            interior=self._interior_stencil / other,
            boundaries={bid: st / other for bid, st in self._boundary_stencils.items()}
        )