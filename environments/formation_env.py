import gym
import matplotlib.pyplot as plt
from gym.spaces import Dict, Discrete, Box, Tuple
import numpy as np
import copy
import networkx as nx


class FormationEnv(gym.Env):
    def __init__(self):
	# Formation Parameter
        formation=2

        # Team parameters and initializations
        #self.n_agents = 3
        self.n_agents = 12

        # state dimension
        self.state_dim = 2

        #self.p = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
        self.p = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 3.0], [-5.0, -1.0], [-1.0, -6.0], [-2.0, -7.0], [-2.0, -9.0], [-8.0, -1.0], [-10.0, -3.0], [-11.0, -2.0], [-4.0, -7.0], [-8.0, -4.0]])
        
        
        self.vel = np.zeros_like(self.p)

        # goal location for leader
        self.leader_goal = np.array([2.0, 2.0])
        self.final_goal = np.array([20.0, 20.0])

        # proportional gain for goal controller
        #self.K_p = 1.0
        #self.K_d = 0.8
        self.K_p = 4.0
        self.K_d = 0.8
        # specify the formation graph
        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.n_agents))
        #self.G.add_edges_from([(0, 1), (1, 2), (2, 0)])
        self.G.add_edges_from([(0, 1), (0, 2), (0, 3), (0,4), (0,5), (0,6), (0,7), (0,8), (0,9), (0,10), (0,11)])

        # simulation time step and timing parameter
        self.iter = 0
        self.max_steps = 200

        self.dt = 0.033/1
        self.done = False

        # env shape   reference formation points
        self.bounds = np.array([-15.0, 25.0, -15.0, 25.0])  # xmin, xmax, ymin, ymax

        #self.formation_ref = np.array([[0.0, 0.0], [1.0, 0.0], [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)]])
        

        if (formation==0):
            self.formation_ref = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0], [6.0, 6.0], [7.0, 7.0], [8.0, 8.0], [9.0, 9.0], [10.0, 10.0], [11.0, 11.0] ])*2
        elif(formation==1):
            self.formation_ref = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 3.0], [3.0, 0.0], [0.0, 5.0], [5.0, 0.0], [0.0, 7.0], [7.0, 0.0], [0.0, 9.0], [9.0, 0.0], [9.0, 9.0] ])*2
        else: self.formation_ref = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [2.0, 1.0], [2.0, 2.0], [2.0, 3.0], [3.0, 1.0], [3.0, 2.0], [4.0, 1.0], [3.0, 3.0] ])*3

        # plotting
        self.fig = None

    def reset(self):

        # initialize robot pose
        self.p = np.array([[0.0, 0.0], [1.0, 0.0], [-2.0, 0.0],[-2.0, 1.0], [-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0]  ])
        self.vel = np.zeros_like(self.p)

        self.iter = 0

        self.last_frame = None
        self.done = False

        self.fig = None


    def step(self, action):

        # propagate goal point
        if self.iter > 100:
            leader_vel = 0.2 * self.K_p * (self.final_goal - self.leader_goal)
            self.leader_goal += self.dt * leader_vel
        else:
            leader_vel = np.zeros_like(self.leader_goal)

        acc = np.zeros((self.n_agents, 2))

        acc[0, :] = 0.5 * self.K_p * (self.leader_goal - self.p[0, :]) + self.K_d * (leader_vel - self.vel[0, :])

        for i in range(1, self.n_agents):
            for j in self.G.neighbors(i):
                acc[i, :] += - self.K_p * ((self.p[i, :] - self.p[j, :]) + (self.formation_ref[i, :] - self.formation_ref[j, :]))


         # TODO: Simplified to first order dynamics!!!
        for i in range(self.n_agents):
            self.vel[i, :] = acc[i, :]
            self.p[i, :] += self.dt * self.vel[i, :]

        observation = self.p

        reward = self.compute_reward()

        self.iter += 1
        done = (self.iter >= self.max_steps)

        return observation, reward, done, {}

    def compute_reward(self):

        reward = 0
        for i in range(self.n_agents):
            for j in self.G.neighbors(i):
                reward += np.linalg.norm(self.p[i, :] - self.p[j, :]) - \
                          np.linalg.norm(self.formation_ref[i, :] - self.formation_ref[j, :])

        return reward

    def render(self):

        if self.fig is None:
            plt.ion()

            # Figure aspect ratio.
            fig_aspect_ratio = 16.0 / 9.0  # Aspect ratio of video.
            fig_pixel_height = 1080  # Height of video in pixels.
            dpi = 300  # Pixels per inch (affects fonts and apparent size of inch-scale objects).

            # Set the figure to obtain aspect ratio and pixel size.
            fig_w = fig_pixel_height / dpi * fig_aspect_ratio  # inches
            fig_h = fig_pixel_height / dpi  # inches
            self.fig, self.ax = plt.subplots(1, 1,
                                             figsize=(fig_w, fig_h),
                                             constrained_layout=True,
                                             dpi=dpi)
            self.ax.set_xlabel('x')
            self.ax.set_ylabel('y')

            data_xlim = [self.bounds[0] - 0.5, self.bounds[1] + 0.5]
            data_ylim = [self.bounds[2] - 0.5, self.bounds[3] + 0.5]

            # Set axes limits which display the workspace nicely.
            self.ax.set_xlim(data_xlim[0], data_xlim[1])
            self.ax.set_ylim(data_ylim[0], data_ylim[1])

            # Setting axis equal should be redundant given figure size and limits,
            # but gives a somewhat better interactive resizing behavior.
            self.ax.set_aspect('equal')

            # Draw robots
            self.robot_handle = self.ax.scatter(self.p[:, 0], self.p[:, 1], 20, 'black')

            self.goal_handle = self.ax.scatter(self.leader_goal[0], self.leader_goal[1], 20, 'red')

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            #self.ax.grid()
            # TODO: Add beautiful renderings for wind!

        else:

            self.robot_handle.set_offsets(self.p)

            self.goal_handle.set_offsets(self.leader_goal)

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
