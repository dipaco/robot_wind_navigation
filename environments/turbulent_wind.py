import gym
import matplotlib.pyplot as plt
from gym.spaces import Dict, Discrete, Box, Tuple
import numpy as np
import copy


class RandomWindEnv(gym.Env):
    def __init__(self, config={}):
        super(RandomWindEnv, self).__init__()

        cx, cy = 0.10, 0.15

        # Final time
        self.action_steps = 10
        tf = 10
        # Timeline
        self.time = np.linspace(0, tf, 240)
        self.Δt = self.time[1] - self.time[0]

        # Desired position
        self.px, self.py = cx * self.time + 1, cy * self.time
        # Desired velocity
        self.vx, self.vy = cx * np.ones(self.time.shape[0]), cy * np.ones(self.time.shape[0])
        # Desired acceleration
        self.ax, self.ay = np.zeros(self.time.shape[0]), np.zeros(self.time.shape[0])

        # setup the action and observation space
        self.config = config
        self.action_space = Box(-5, 5, (2,))
        self.observation_space = Box(-5, 5, (4,))

        # The SAC training script requires this attribute
        self._max_episode_steps = tf

        self.reset()

    def reset(self):

        # Initial conditions
        self.x = np.array([1, 0, 0., 0.])
        self.dx = np.zeros(4)

        # Random disturbance
        self.Dx = np.random.normal(3, 5, self.time.shape[0])
        self.Dy = np.random.normal(0, 5, self.time.shape[0])

        # PD gains
        self.kp, self.kd = 5, 2

        self.x_log = np.array([copy.copy(self.x)])  # log the robot state

        self.currrent_step = 1

        # For rendering purposes
        self.last_frame = None

        '''return np.concatenate([
            [self.px[self.currrent_step - 1], self.py[self.currrent_step - 1]],
            [self.vx[self.currrent_step - 1], self.vy[self.currrent_step - 1]],
            self.x])'''
        return np.array([
            self.px[self.currrent_step - 1] - self.x[0], self.py[self.currrent_step - 1] - self.x[1],
            self.vx[self.currrent_step - 1] - self.x[2], self.vy[self.currrent_step - 1] - self.x[3]])

    # Euler integration
    def _simulate(self, Δt, x, u):
        x += Δt * u
        return x

    def step(self, action):
        if self.currrent_step < self.time.shape[0]:

            # Control input
            # FIXME: Just adding the action to the control is probably wrong, I should change this ASAP
            tk = self.currrent_step
            '''u = [self.kp * (self.px[tk] - self.x[0]) + self.kd * (self.vx[tk] - self.x[2]) + self.ax[tk] + self.Dx[tk] + action[0],
                 self.kp * (self.py[tk] - self.x[1]) + self.kd * (self.vy[tk] - self.x[3]) + self.ay[tk] + self.Dy[tk] + action[1]]#'''
            '''u = [self.kp * (self.px[tk] - self.x[0]) + self.kd * (self.vx[tk] - self.x[2]) + self.ax[tk] + self.Dx[tk] + action[0],
                 self.kp * (self.py[tk] - self.x[1]) + self.kd * (self.vy[tk] - self.x[3]) + self.ay[tk] + self.Dy[tk] + action[1]]
'''
            u = [self.kd * (self.vx[tk] - self.x[2]) + self.ax[tk] + self.Dx[tk] + action[0],
                 self.kd * (self.vy[tk] - self.x[3]) + self.ay[tk] + self.Dy[tk] + action[1]]

            #u = self.action_space.low.min() + 0.5 * (self.action_space.low.max() - self.action_space.low.min()) * (action + 1)

            u = action

            # Dynamics
            self.dx[:2] = self.x[2:]
            self.dx[2:] = u

            self.x = self._simulate(self.Δt, self.x, self.dx)

            self.x_log = np.concatenate([self.x_log, [copy.copy(self.x)]])

            reward = -0.1 * np.linalg.norm(np.stack([self.px, self.py])[:, self.currrent_step] - self.x[:2])
            #reward += -np.linalg.norm(np.stack([self.vx, self.vy])[:, self.currrent_step] - self.x[2:])


            #final_pos_eps = 2
            #if np.linalg.norm(np.stack([self.px, self.py])[:, -1] - self.x[:2]) <= final_pos_eps:
            #    reward += 2000

            self.currrent_step += 1
        else:
            reward = 0

        '''observation = np.concatenate([
            [self.px[self.currrent_step - 1], self.py[self.currrent_step - 1]],
            [self.vx[self.currrent_step - 1], self.vy[self.currrent_step - 1]],
            self.x])'''
        observation = np.array([
            self.px[self.currrent_step - 1] - self.x[0], self.py[self.currrent_step - 1] - self.x[1],
            self.vx[self.currrent_step - 1] - self.x[2], self.vy[self.currrent_step - 1] - self.x[3]])

        if not (-5 <= self.x[0] <= 5 and -5 <= self.x[1] <= 5):
            reward -= 100
            done = True
        elif self.currrent_step >= self.time.size:
            done = True
        else:
            done = False

        return (
            observation,
            reward,
            done,
            {},
        )

    def render(self, mode='rgb'):

        if self.currrent_step % 3 == 0 or self.last_frame is None:

            fig = plt.figure()

            # Plot trajectory and velocity,
            plt.plot(self.px, self.py)
            plt.plot(self.px[self.currrent_step - 1], self.py[self.currrent_step - 1], 'bx')
            # quiver(px, py, vx, vy)

            # starting point
            plt.plot(self.px[0], self.py[0], 'ro')

            # Desired Path
            plt.plot(self.px, self.py, 'b--')
            # Path
            plt.plot(self.x_log[:, 0], self.x_log[:, 1], 'r--')
            plt.plot(self.x_log[-1, 0], self.x_log[-1, 1], 'ro')
            plt.title(f'ax: {self.dx[2]:.2f}, ay: {self.dx[3]:.2f}')

            # plot bounds
            plt.xlim(-5, 5)
            plt.ylim(-5, 5)

            fig.canvas.draw()
            image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            plt.close(fig)

            self.last_frame = image_from_plot

        return self.last_frame


class TurbulentWindEnv(gym.Env):
    def __init__(self, config={}):
        super(TurbulentWindEnv, self).__init__()

        self.config = config
        self.n = 2  # number of robots in the formation (nodes)
        self.m = 3
        self.action_space = Box(-1, 1, (self.n, 2))
        self.observation_space = Box(-1, 1, (self.n, self.m))
        self.p_done = config.get("p_done", 0.1)

    def reset(self):
        return self.observation_space.sample()

    def step(self, action):

        chosen_action = action[0]
        cnt_control = action[1][chosen_action]

        if chosen_action == 0:
            reward = cnt_control
        else:
            reward = -cnt_control - 1

        print(f"Action, {chosen_action} continuous ctrl {cnt_control}")
        return (
            self.observation_space.sample(),
            reward,
            bool(np.random.choice([True, False], p=[self.p_done, 1.0 - self.p_done])),
            {},
        )