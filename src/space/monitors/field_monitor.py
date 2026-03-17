from space.field import FieldView
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


class FieldMonitor:
    def __init__(self, field: FieldView, save_every_n_steps: int = 1):
        if len(field.shape) != 3:
            raise ValueError("Field monitor only handles 2D data plots.")

        self._field = field
        self._save_every_n_steps = save_every_n_steps
        self._values = []
        self._time = field.space.time
        self._time.register_advanceable(self)

    def advance(self, dt: float):
        self._values.append(self._field.value().eval())

    def playback(self):
        dts = self._time._dts
        values = np.array(self._values)

        x, y = self._field.space.discretization.points()
        times = np.cumsum(dts[: len(values)])

        if self._field.components == 1:
            data = values[:, 0]
        else:
            data = np.linalg.norm(values, axis=1)

        fig, ax = plt.subplots(figsize=(8, 6))
        vmin = data.min()
        vmax = data.max()
        contour = ax.contourf(
            x, y, data[0], levels=50, vmin=vmin, vmax=vmax, cmap="viridis"
        )
        cbar = fig.colorbar(contour)
        title = ax.set_title(f"t = {times[0]:.3f}")

        def update(frame):
            ax.clear()
            contour = ax.contourf(
                x, y, data[frame], levels=50, vmin=vmin, vmax=vmax, cmap="viridis"
            )

            ax.set_title(f"t = {times[frame]:.3f}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        anim = FuncAnimation(fig, update, frames=len(data), interval=100)

        plt.tight_layout()
        plt.show()