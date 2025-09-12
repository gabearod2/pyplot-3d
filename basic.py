# basic.py
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Line:
    """Lightweight 3D line that can be updated with set_data_3d."""
    def __init__(self, ax, c='k', linewidth=1.5):
        self.ax = ax
        (self.artist,) = ax.plot([], [], [], color=c, linewidth=linewidth)

    def draw_from_to(self, x0, x1):
        self.artist.set_data_3d([x0[0], x1[0]],
                                [x0[1], x1[1]],
                                [x0[2], x1[2]])


class Arrow:
    """Arrow rendered as a line (much faster & easier to update than quiver)."""
    def __init__(self, ax, c='r', length=1.0, linewidth=2.0):
        self.ax = ax
        self.length = length
        (self.artist,) = ax.plot([], [], [], color=c, linewidth=linewidth)

    def draw_from_to(self, x, u):
        tip = x + u  # u already encodes direction & scale
        self.artist.set_data_3d([x[0], tip[0]],
                                [x[1], tip[1]],
                                [x[2], tip[2]])


class Sphere:
    """Center marker for the UAV (fast point instead of heavy surface)."""
    def __init__(self, ax, r, c='k'):
        self.ax = ax
        # Use a marker; markersize is in points, scale empirically vs r
        (self.artist,) = ax.plot([], [], [], marker='o', color=c,
                                 markersize=max(1, int(200 * r)))

    def draw_at(self, position):
        self.artist.set_data_3d([float(position[0])],
                                [float(position[1])],
                                [float(position[2])])


class Plate:
    """
    Filled rotor disc using Poly3DCollection.
    Create once; update vertices each frame by rotating/translating a cached circle.
    """
    def __init__(self, ax, r, c='k', x=np.array([0, 0, 0.0]), R=np.eye(3), resolution=60, alpha=0.9):
        self.ax = ax
        self.r = r
        self.color = c
        self.alpha = alpha

        theta = np.linspace(0.0, 2*np.pi, resolution, endpoint=True)
        self.circle_local = np.vstack([r*np.cos(theta), r*np.sin(theta), np.zeros_like(theta)])  # (3,M)

        # Create a placeholder polygon and add to axes
        self.collection = Poly3DCollection([[(0.0, 0.0, 0.0)]],
                                           facecolor=self.color,
                                           edgecolor=self.color,
                                           linewidth=0.6,
                                           alpha=self.alpha)
        ax.add_collection3d(self.collection)

        # Initialize at provided pose if desired
        self.draw_at(x, R)

    def draw_at(self, x, R):
        pts = (R @ self.circle_local) + x[:, None]  # (3, M)
        verts = [tuple(p) for p in pts.T]           # list of (x,y,z)
        self.collection.set_verts([verts])
