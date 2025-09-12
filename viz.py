
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from flightning.pyplot3d.uav import Uav 
from flightning.envs.env_base import EnvTransition
from flightning.envs.env_base import EnvState


class QuadrotorAnimator():
    def __init__(
            self,
            world_box_min,
            world_box_max,
            goal_x,
            goal_R
            ):
        
        plt.style.use('seaborn-v0_8')
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')

        mins = np.asarray(world_box_min)
        maxs = np.asarray(world_box_max)
        self.xmin, self.xmax = mins[0]/2, maxs[0]/2
        self.ymin, self.ymax = mins[1]/2, maxs[1]/2
        self.zmin, self.zmax = mins[2]/2, maxs[2]/2
        self.ax.set_xlim([self.xmin, self.xmax])
        self.ax.set_ylim([self.ymin, self.ymax])
        self.ax.set_zlim([self.zmin, self.zmax])

        self.goal_x = np.asarray(goal_x)
        self.goal_R = np.asarray(goal_R)

    def animate_trajectories(
            self, 
            traj: EnvTransition,
            filename: str
        ):
        assert traj.reward.ndim == 2
        num_trajs = traj.reward.shape[0]
        state: EnvState = traj.state
        done = np.logical_or(traj.terminated, traj.truncated)
        idx = np.where(done[0])[0][0].item() + 1

        uavs = [Uav(self.ax) for _ in range(num_trajs)]
        goal_uav = Uav(self.ax, color='c')

        x = np.zeros((num_trajs, idx, 3)) # (20, 501, 3) traj, idx, pos
        R = np.zeros((num_trajs, idx, 3, 3)) # (20, 501, 3, 3) traj, idx, R

        for i in range(num_trajs):
            idx = np.where(done[i])[0][0].item() + 1
            x[i, :idx, :] = state.quadrotor_state.p[i, :idx, 0:3] 
            R[i, :idx, :] = state.quadrotor_state.R[i, :idx]

        interval = (state.time[0, 1] - state.time[0, 0])*1000 

        ani = FuncAnimation(
            self.fig, 
            self.update_plot, 
            frames=idx, 
            fargs=(x, R, uavs, goal_uav), 
            interval=interval
        )
        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()
        plt.title("Quadrotor Animation (Press 'Q' to Exit)")
        if filename is not None: 
            from matplotlib.animation import PillowWriter
            fps = max(1, int(round(1000.0 / float(interval))))  # interval in ms
            ani.save(filename=filename, writer=PillowWriter(fps=fps), dpi=200)
        plt.show()


    def update_plot(self, i, x, R, uavs, goal_uav):

        for k, uav in enumerate(uavs):
            uav.draw_at(x[k, i, :], R[k, i, :, :])
        goal_uav.draw_at(self.goal_x, self.goal_R)
        return []
