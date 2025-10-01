import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib import patheffects as pe
from pathlib import Path

from flightning.pyplot3d.uav import Uav 
from flightning.envs.env_base import EnvTransition, EnvState


class QuadrotorAnimator():
    def __init__(
            self,
            world_box_min,
            world_box_max,
            goal_x,
            goal_R
            ):
        # Base style (nice defaults for interactive preview)
        plt.style.use('seaborn-v0_8')
        plt.rcParams.update({
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
        })

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')

        mins = np.asarray(world_box_min)
        maxs = np.asarray(world_box_max)
        self.xmin, self.xmax = mins[0], maxs[0]
        self.ymin, self.ymax = mins[1], maxs[1]
        self.zmin, self.zmax = mins[2], maxs[2]
        self.ax.set_xlim([self.xmin, self.xmax])
        self.ax.set_ylim([self.ymin, self.ymax])
        self.ax.set_zlim([self.zmin, self.zmax])

        # Keep axes labels subtle and consistent
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        self.ax.set_zlabel("z [m]")

        # Orthographic projection looks crisper for engineering plots
        try:
            self.ax.set_proj_type('ortho')
        except Exception:
            pass

        # aspect ratio = world box
        try:
            self.ax.set_box_aspect((self.xmax-self.xmin,
                                    self.ymax-self.ymin,
                                    self.zmax-self.zmin))
        except Exception:
            pass

        # panes
        self.ax.grid(False)
        for axis in [self.ax.xaxis, self.ax.yaxis, self.ax.zaxis]:
            try:
                axis.pane.set_facecolor((1,1,1,0))  
                axis.pane.set_edgecolor((0,0,0,0.2))
            except Exception:
                pass

        self.goal_x = np.asarray(goal_x)
        self.goal_R = np.asarray(goal_R)

    def animate_trajectories(
            self, 
            traj: EnvTransition,
            filename: str = None,
            dpi: int = 200,
            bitrate: int = 4000,
            title: str = "Quadrotor Animation",
            show_hud: bool = True
        ):
        assert traj.reward.ndim == 2
        num_trajs = traj.reward.shape[0]
        state: EnvState = traj.state

        uavs = [Uav(self.ax) for _ in range(num_trajs)]
        goal_uav = Uav(self.ax, color='c')

        # first termination index (per env) + 1
        idxs = np.zeros(num_trajs, dtype=np.int32)
        done = np.logical_or(traj.terminated, traj.truncated)
        for i in range(num_trajs):
            idxs[i] = int(np.where(done[i])[0][0]) + 1

        n_frames = int(np.max(idxs))

        # (num_trajs, n_frames, 3/3x3)
        x = np.zeros((num_trajs, n_frames, 3))
        R = np.zeros((num_trajs, n_frames, 3, 3))

        for j in range(num_trajs):
            idx = int(idxs[j])
            x[j, :idx, :] = state.quadrotor_state.p[j, :idx, 0:3]
            R[j, :idx, :] = state.quadrotor_state.R[j, :idx]

        interval = float(state.time[0, 1] - state.time[0, 0]) * 1000.0  # ms
        fps = max(1, int(round(1000.0 / interval)))

        # Header title
        self.fig.suptitle(title, fontweight="bold", y=0.98)

        hud_artist = None
        if (not filename) and show_hud:
            # Subtle hint for interactive mode only (not exported)
            hud_artist = self.fig.text(0.012, 0.972, "Press Q to exit",
                                       ha="left", va="top", alpha=0.7,
                                       fontsize=10)
            hud_artist.set_path_effects([pe.withStroke(linewidth=2, foreground="white")])

        ani = FuncAnimation(
            self.fig, 
            self.update_plot, 
            frames=range(n_frames), 
            fargs=(x, R, uavs, goal_uav), 
            interval=interval,
            blit=False  # blitting is not supported for 3D artists
        )

        # Avoid fullscreen changes during export
        try:
            if not filename:
                manager = plt.get_current_fig_manager()
                manager.full_screen_toggle()
        except Exception:
            pass

        # Save if a filename is provided. Choose writer by extension.
        if filename:
            # Neutral style for export, then re-apply typography so fonts match
            plt.style.use('default')
            plt.rcParams.update({
                "font.family": "DejaVu Sans",
                "font.size": 12,
                "axes.titlesize": 14,
                "axes.labelsize": 12,
            })

            path = Path(filename)
            ext = path.suffix.lower()

            if ext in {'.mp4', '.m4v', '.mov'}:
                writer = FFMpegWriter(fps=fps, codec='libx264', bitrate=bitrate, extra_args=['-pix_fmt','yuv420p'])
            elif ext == '.gif':
                writer = PillowWriter(fps=fps)
            else:
                writer = FFMpegWriter(fps=fps, codec='libx264', bitrate=bitrate, extra_args=['-pix_fmt','yuv420p'])
                path = path.with_suffix('.mp4')

            # Optional footer for exports
            self.fig.text(0.01, 0.01, "AcroRL", fontsize=9, alpha=0.6)

            ani.save(str(path), writer=writer, dpi=dpi)

        plt.show()

    def update_plot(self, i, x, R, uavs, goal_uav):
        for k, uav in enumerate(uavs):
            uav.draw_at(x[k, i, :], R[k, i, :, :])
        goal_uav.draw_at(self.goal_x, self.goal_R)
        return []
