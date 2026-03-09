from __future__ import annotations
import copy
from typing import Self
import numpy as np

import algebra
from algebra.exceptions import ShapeMismatchException
from algebra.stencil import Stencil
from algebra.expression import ScalarExpression

import tools.region as region
from discr.fd.domain import FDDomain
from discr.core.domain import BoundaryId

class FDStencilOperator(algebra.Operator):
    def __init__(self, domain: FDDomain, stencil: Stencil):
        super().__init__(domain.grid.shape, domain.grid.shape)

        self._domain = domain
        self._interior_stencil = stencil

        self._boundary_stencils: dict[BoundaryId, Stencil] = {}
        for bid in domain.boundaries:
            self._boundary_stencils[bid] = stencil.copy()

        self._factor: ScalarExpression = ScalarExpression(lambda: 1.0)

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

    def resolve_factor(self):
        factor = self._factor.eval() 
        for bid, stencil in self._boundary_stencils.items():
            self._boundary_stencils[bid] = stencil * factor
        self._interior_stencil *= factor
        self._factor = ScalarExpression(lambda: 1.0)

    def copy(self) -> FDStencilOperator:
        return self._new_copy(
            self.interior_stencil, self.boundary_stencils, self._factor
        )

    def _apply(self, field: np.ndarray, out: np.ndarray):
        interior = region.interior(
            field.shape,
            tuple(self._interior_stencil.ax_ranges().values()),
        )
        # interior = self.domain.grid.interior

        self._interior_stencil.apply_to_region(field, out, interior)

        for bid, stencil in self._boundary_stencils.items():
            boundary = self._domain.boundary(bid)
            stencil.apply_to_region(field, out, boundary.region)
        out *= self._factor.eval()

    def _new_copy(
        self,
        interior: Stencil,
        boundaries: dict[BoundaryId, Stencil],
        factor: ScalarExpression
    ) -> Self:
        op = FDStencilOperator(self.domain, copy.deepcopy(interior))
        op._boundary_stencils = copy.deepcopy(boundaries)
        op._factor = copy.deepcopy(factor)
        return op

    def __neg__(self) -> Self:
        return self._new_copy(
            interior=-self._interior_stencil,
            boundaries={bid: -st for bid, st in self._boundary_stencils.items()},
            factor=self._factor,
        )

    def __add__(self, other: FDStencilOperator) -> Self:
        if not isinstance(other, FDStencilOperator):
            return NotImplemented
        if (
            self.input_shape != other.input_shape
            or self.output_shape != other.output_shape
        ):
            raise ShapeMismatchException("Operator shape mismatch")
        self.resolve_factor()
        other.resolve_factor()
        return self._new_copy(
            self._interior_stencil + other._interior_stencil,
            {
                bid: self._boundary_stencils[bid]
                + other._boundary_stencils[bid]
                for bid in self._boundary_stencils
            },
            ScalarExpression(lambda: 1.0)
        )

    def __mul__(self, other: float | ScalarExpression) -> Self:
        if isinstance(other, (float,ScalarExpression)):
            return self._new_copy(
                self._interior_stencil,
                self._boundary_stencils,
                self._factor * other
            )
        return NotImplemented

    def __rmul__(self, other: float | ScalarExpression) -> Self:
        return self * other

    def __truediv__(self, other: float | ScalarExpression) -> Self:
        if isinstance(other, float):
            self._interior_stencil += other
            for bid, in self._boundary_stencils.keys():
                self._boundary_stencils[bid] /= other
        if isinstance(other, ScalarExpression):
            return self._new_copy(
                self._interior_stencil,
                self._boundary_stencils,
                self._factor / other
            )
        return NotImplemented