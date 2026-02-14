import numbers
import numpy as np

class FDStencil():
    def __init__(self, contrib: dict[int, dict[int, float]]):
        self._contrib = contrib 

    def ax_contrib(self, ax: int) -> dict[int, float]:
        return self._contrib.get(ax, {})

    def ax_contrib_range(self, ax: int):
        return self.ax_contrib(ax).keys()

    def interior_region(self, shape: tuple[int, ...]) -> tuple[slice, ...]:
        region = []

        for ax in range(len(shape)):
            offsets = self.ax_contrib_range(ax)
            if not offsets:
                region.append(slice(None))
                continue

            max_off = max(abs(k) for k in offsets)
            region.append(slice(max_off, shape[ax] - max_off))

        return tuple(region)

    def boundary_region(
        self, shape: tuple[int, ...], ax: int, inward_dir: int
    ) -> tuple[slice, ...]:

        region = [slice(None)] * len(shape)

        if inward_dir == 1:
            region[ax] = slice(0, 1)
        else:
            region[ax] = slice(shape[ax] - 1, shape[ax])

        return tuple(region)

    @staticmethod
    def shift_region(
        region: tuple[slice, ...], ax: int, k: int
    ) -> tuple[slice, ...]:

        shifted = list(region)
        s = region[ax]

        shifted[ax] = slice(
            s.start + k,
            s.stop + k
        )

        return tuple(shifted)

    def apply_to_region(
        self, field: np.ndarray, out: np.ndarray, region: tuple[slice, ...],
    ):
        for ax in range(field.ndim):
            contrib = self.ax_contrib(ax)

            for k, c in contrib.items():
                src_region = self.shift_region(region, ax, k)
                out[region] += c * field[src_region]
    def apply(
        self, field: np.ndarray, out: np.ndarray,
    ):
        region = self.interior_region(field.shape)
        self.apply_to_region(field, out, region)

    def copy(self) -> "FDStencil":
        new = FDStencil()
        new._contrib = {
            ax: dict(offsets)
            for ax, offsets in self._contrib.items()
        }
        return new

    def _combine(self, other: "FDStencil", op):
        result = self.copy()

        for ax, offsets in other._contrib.items():
            if ax not in result._contrib:
                result._contrib[ax] = {}

            for k, c in offsets.items():
                if k in result._contrib[ax]:
                    result._contrib[ax][k] = op(
                        result._contrib[ax][k], c
                    )
                else:
                    result._contrib[ax][k] = op(0.0, c)

        return result

    def __add__(self, other):
        if not isinstance(other, FDStencil):
            return NotImplemented
        return self._combine(other, lambda a, b: a + b)

    def __iadd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if not isinstance(other, FDStencil):
            return NotImplemented
        return self._combine(other, lambda a, b: a - b)

    def __isub__(self, other):
        return self.__sub__(other)

    def __neg__(self):
        result = self.copy()
        for ax in result._contrib:
            for k in result._contrib[ax]:
                result._contrib[ax][k] *= -1.0
        return result

    def _scale(self, scalar: float):
        result = self.copy()
        for ax in result._contrib:
            for k in result._contrib[ax]:
                result._contrib[ax][k] *= scalar
        return result

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            return self._scale(other)

        if isinstance(other, FDStencil):
            result = FDStencil()

            for ax1, offsets1 in self._contrib.items():
                for ax2, offsets2 in other._contrib.items():
                    if ax1 != ax2:
                        raise NotImplementedError(
                            "Stencil multiplication only supported along same axis."
                        )

                    if ax1 not in result._contrib:
                        result._contrib[ax1] = {}

                    for k1, c1 in offsets1.items():
                        for k2, c2 in offsets2.items():
                            k = k1 + k2
                            result._contrib[ax1][k] = (
                                result._contrib[ax1].get(k, 0.0)
                                + c1 * c2
                            )

            return result

        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, scalar):
        if not isinstance(scalar, numbers.Number):
            return NotImplemented
        return self._scale(1.0 / scalar)

    def __itruediv__(self, scalar):
        return self.__truediv__(scalar)