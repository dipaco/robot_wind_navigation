import math
import os
import sys
import gym
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import copy
import networkx as nx
import netCDF4 as nc
import torch
from gym.spaces import Dict, Discrete, Box, Tuple
from scipy.interpolate import interp2d
from scipy.interpolate import griddata
from torch.nn.functional import grid_sample

__BASE_FOLDER__ = os.path.dirname(os.path.abspath(__file__))
dist_package_folder = os.path.join(__BASE_FOLDER__, '../gcnn_agents/')
sys.path.append(dist_package_folder)
from agent_utils import get_formation_conf, normalize


class TurbulentFormationEnv(gym.Env):

    def __init__(self, config):

        # setup the config options
        self.config = config

        # Set the wind sensor
        self.WIND_SENSOR_MAX_SPEED = self.config.wind_sensor.max_speed
        self.WIND_SENSOR_MAG_ACC = self.config.wind_sensor.speed_accuracy #0.02 * self.WIND_SENSOR_MAX_SPEED  # * 0.00005
        self.WIND_SENSOR_DIR_ACC = self.config.wind_sensor.angle_accuracy * np.pi / 180 #3 * np.pi / 180  # * 0.00005
        self.WIND_SENSOR_MAG_RES = 1  # number of decimal places
        self.WIND_SENSOR_DIR_RES = 2  # number of decimal places

        # controls whether we want to start the simulation from time=0 or from some offset
        self.sim_time_offset = 15.0

        # Quadracopters info: https://funoffline.com/how-much-does-a-drone-weigh/

        # Wind simulation constants
        # air density at 25 Cº https://www.engineeringtoolbox.com/air-density-specific-weight-d_600.html?vA=30&units=C#
        self.rho = 1.184
        self.P_s = 101325   # 1 atm in Pascal units
        # mass in Kg
        self.m = 0.3
        # reference area of a sphere or r=10 cm
        self.robot_radius = 0.05
        self.area = 4 * np.pi * (self.robot_radius ** 2)
        # drag coefficient of a sphere
        self.c_d = 0.47

        # env shape   reference formation points
        self.bounds = np.array([-5.0, 5.0, -5.0, 5.0])  # xmin, xmax, ymin, ymax

        self.MAX_WIND_SIMULATION_SPEED = 15.0 # 17.0 ~ 60 km/h
        self.MAX_ROBOT_ACCELERATION = 0.5 * self.rho * self.c_d * self.area * (self.MAX_WIND_SIMULATION_SPEED ** 2) / self.m

        # Team parameters and initializations
        self.n_agents = self.config.formation_params.num_nodes

        # action dim
        self.action_dim = 2

        # state dimension
        self.state_dim = 4 * self.action_dim
        if self.config.use_wind_pressure_sensors:
            self.state_dim += self.action_dim + 2 # 2 -> pressure and wind magnitude

        self.G = None
        self.formation_ref = None
        self.formation_ref, self.G = get_formation_conf(self.config.formation_params.formation_type, robot_distance=self.config.formation_params.formation_radius)

        # initialize robot pose
        if self.config.formation_params.init_points == 'random':
            self.p = 0.8 * (self.bounds[1] - self.bounds[0]) * np.random.rand(self.n_agents, 2) + self.bounds[0]
            self.leader_goal = 0.7 * (self.bounds[1] - self.bounds[0]) * np.random.rand(2) + self.bounds[0]
        elif self.config.formation_params.init_points == 'in_formation':
            randon_translation = 0.3 * (self.bounds[1] - self.bounds[0]) * (2 * np.random.rand(1, 2) - 1)
            self.p = np.copy(self.formation_ref) + randon_translation
            self.leader_goal = self.p[0, :]
        else:
            raise ValueError(f'Wrong point initialization "{self.config.formation_params.init_points}". Try [random, in_formation]')

        # self.p = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])

        self.vel = np.zeros_like(self.p)

        # goal location for leader
        #self.leader_goal = np.array([2.0, 2.0])

        self.final_goal = np.array([0.0, 0.0])

        # proportional gain for goal controller
        self.K_p = 2.5  # 1.0*5
        self.K_d = 1.25 # 0.5*5

        # simulation time step and timing parameter
        self.iter = 0
        self._max_episode_steps = 900 # 450

        self.dt = 0.066
        self.done = False

        # plotting
        self.fig = None

        # Metrics
        self.formation_error = np.zeros((self.n_agents, self._max_episode_steps))
        self.position_error = np.zeros((self.n_agents, self._max_episode_steps))

        # setup turbulence
        if self.config.turbulence_model == 'random':
            self.setup_random_turbulence()
        elif self.config.turbulence_model == 'NS':
            self.wind_sim_dict = {}
            self.selected_sim = None
            # all_sims = [d for d in map(lambda x: os.path.join(sim_path, x), os.listdir(sim_path)) if os.path.isdir(d)]
            self.all_sims = [d for d in os.listdir(self.config.turbulence_base_folder) if os.path.isdir(os.path.join(self.config.turbulence_base_folder, d))]
            assert len(self.all_sims) > 0, f"There are no valid simulation in {self.config.turbulence_base_folder}"

        elif self.config.turbulence_model == 'constant':
            self.wind_theta = np.pi * (2*np.random.rand() - 1)
            self.wind_theta = 0.0

        # setup the action and observation space
        self.action_space = Box(-1.0, 1.0, (self.n_agents * self.action_dim,))
        self.observation_space = Box(self.bounds[0], self.bounds[1], (self.n_agents * self.state_dim,))

        #self.last_action = None

    def setup_random_turbulence(self):
        self.num_dist_seeds = 5
        self.turbulence_seeds = np.random.rand(self.num_dist_seeds, 2) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        self.direction = np.sign(np.random.rand(self.num_dist_seeds) - 0.5)

    def _setup_NS_turbulence(self):

        sim_idx = np.random.randint(len(self.all_sims))
        self.selected_sim = self.all_sims[sim_idx]

        if self.selected_sim not in self.wind_sim_dict:

            self.wind_sim_dict[self.selected_sim] = {}

            sim_folder = os.path.join(self.config.turbulence_base_folder, self.selected_sim)
            # finds all occurrences of the substring '_'
            idx_ = [i for i in range(len(self.selected_sim)) if self.selected_sim.startswith('_', i)]
            res_x, res_y = map(int, self.selected_sim[idx_[0] + 1:idx_[1]].split('x'))

            sim_files = [os.path.join(sim_folder, f) for f in os.listdir(sim_folder) if f.endswith('.nc')]
            sim_files.sort()

            num_t = len(sim_files)
            assert num_t > 0, f"The simulation {selected_sim} does not contain any data file (*.nc)"

            self.wind_sim_dict[self.selected_sim]['v'] = np.zeros((num_t, res_x, res_y, 2))
            self.wind_sim_dict[self.selected_sim]['p'] = np.zeros((num_t, res_x, res_y))

            # TODO: This would probably be cleaner with a regular expression
            init_token = 'state_phys_t'
            end_token = '.nc'
            self.wind_sim_dict[self.selected_sim]['t'] = np.array(
                [float(t[t.find(init_token) + len(init_token):t.find(end_token)]) for t in sim_files]
            )

            for i, f in enumerate(sim_files):
                ds = nc.Dataset(f)
                #print(ds.groups['state_phys']['ux'][:].shape)
                self.wind_sim_dict[self.selected_sim]['v'][i, :, :, 0] = ds.groups['state_phys']['ux'][:]
                self.wind_sim_dict[self.selected_sim]['v'][i, :, :, 1] = ds.groups['state_phys']['uy'][:]

    def reset(self):

        # initialize robot pose
        if self.config.formation_params.init_points == 'random':
            self.p = 0.8 * (self.bounds[1] - self.bounds[0]) * np.random.rand(self.n_agents, 2) + self.bounds[0]
            self.leader_goal = 0.7 * (self.bounds[1] - self.bounds[0]) * np.random.rand(2) + self.bounds[0]
        elif self.config.formation_params.init_points == 'in_formation':
            randon_translation = 0.3 * (self.bounds[1] - self.bounds[0]) * (2 * np.random.rand(1, 2) - 1)
            self.p = np.copy(self.formation_ref) + randon_translation
            self.leader_goal = self.p[0, :]
        else:
            raise ValueError(f'Wrong point initialization "{self.config.formation_params.init_points}". Try [random, in_formation]')

        #self.p = np.array([[0.0, 0.0], [1.0, 0.0], [-2.0, 0.0]])

        self.vel = np.zeros_like(self.p)

        self.iter = 0

        self.last_frame = None
        self.done = False

        self.fig = None

        # setup the turbpulence
        if self.config.turbulence_model == 'random':
            self.setup_random_turbulence()
        elif self.config.turbulence_model == 'NS':
            self._setup_NS_turbulence()

        if self.config.use_wind_pressure_sensors:

            # Get the wind velocity at the query points
            if self.config.turbulence_model is not None:
                v_we = self.get_disturbance(self.p)
                # Velocity of the wind with respect to the robot
                v_wr = v_we - self.vel
            else:
                v_wr = np.zeros_like(self.vel)

            # Get the total pressure
            P_t = self._get_pressure(v_wr)

            # Observations are a flatten version of the state matrix
            if self.config.measurement_noise:
                v_wr_sensor = self._real_wind_to_sensor(v_wr)
            else:
                v_wr_sensor = v_wr

            # computes the squared wind magnitude
            v_wr_mag = (v_wr_sensor ** 2).sum(axis=-1, keepdims=True)

            observation = np.concatenate([
                self.p, self.vel, v_wr_sensor, P_t, v_wr_mag, np.zeros([self.n_agents, 2 * self.action_dim])
            ], axis=1).reshape(-1)
        else:
            observation = np.concatenate([
                self.p, self.vel, np.zeros([self.n_agents, 2 * self.action_dim])
            ], axis=1).reshape(-1)

        return observation

    def _real_wind_to_sensor(self, w_real):
        """
            Simulates an ultrasonic wind sensor and its inaccuracies.
        """
        # We add noise to the measurement to simulate sensor accuracy/uncertainty
        noise_in_polars = np.random.normal(loc=np.zeros_like(w_real), scale=[self.WIND_SENSOR_MAG_ACC, self.WIND_SENSOR_DIR_ACC] * np.ones_like(w_real))
        w_sensed = self._polar2cartesian(self._cartesian2polar(w_real) + noise_in_polars)

        # We round the numbers to simulate sensor resolution
        w_sensed = np.stack([
            np.round(w_sensed[..., 0], self.WIND_SENSOR_MAG_RES),
            np.round(w_sensed[..., 1], self.WIND_SENSOR_DIR_RES),
        ], axis=-1)

        return w_sensed

    def _cartesian2polar(self, v):
        assert v.shape[-1] == 2, 'Can only convert to polar vectors of dimension 2.'
        mag = np.linalg.norm(v, axis=-1)
        theta = np.arctan2(v[..., 1], v[..., 0])
        return np.stack([mag, theta], axis=-1)

    def _polar2cartesian(self, v):
        assert v.shape[-1] == 2, 'Can only convert from polar vectors of dimension 2.'
        x = np.stack([
            v[..., 0] * np.cos(v[..., 1]),
            v[..., 0] * np.sin(v[..., 1]),
        ], axis=-1)
        return x

    def _get_pressure(self, v_wr):
        # Get the pressure at the robot's locations
        P_d = 0.5 * self.rho * (v_wr ** 2).sum(axis=1)
        # Computes the total pressure
        #P_t = P_d + self.P_s

        return P_d[:, None]

    def step(self, action):
        # Reshape and scale the action from the controller
        action = self.MAX_ROBOT_ACCELERATION * action.reshape(self.n_agents, -1)

        self.last_action = np.copy(action)

        # propagate goal point
        #if self.iter > 0:
        #    leader_vel = 0.2 * self.K_p * (self.final_goal - self.leader_goal)
        #    self.leader_goal += self.dt * leader_vel
        #else:
        #    leader_vel = np.zeros_like(self.leader_goal)
        leader_vel = np.zeros_like(self.leader_goal)

        acc = np.zeros((self.n_agents, 2))
        pd_acc = np.zeros((self.n_agents, 2))

        #acc[0, :] = 0.5 * self.K_p * (self.leader_goal - self.p[0, :]) + self.K_d * (leader_vel - self.vel[0, :])
        pd_acc[0, :] = self.K_p * (self.leader_goal - self.p[0, :]) + self.K_d * (leader_vel - self.vel[0, :])

        # Applies the PD control
        if self.config.formation_params.formation_contro_type == 'simple':
            p_d = self.leader_goal + self.formation_ref
            v_d = np.zeros_like(self.vel)
            pd_acc = self.K_p * (p_d - self.p) + self.K_d * (v_d - self.vel)
        elif self.config.formation_params.formation_contro_type == 'magnus':
            for i in range(1, self.n_agents):
                for j in self.G.neighbors(i):
                    # Applies the formation control action
                    pd_acc[i, :] += - self.K_p * ((self.p[i, :] - self.p[j, :]) + (self.formation_ref[i, :] - self.formation_ref[j, :])) - self.K_d * (self.vel[i, :] - self.vel[j, :])

        acc += pd_acc

        # Get the wind velocity at the query points
        if self.config.turbulence_model is not None:
            v_we = self.get_disturbance(self.p)
            # Velocity of the wind with respect to the robot
            v_wr = v_we - self.vel
        else:
            v_wr = np.zeros_like(self.vel)

        # Applies the action from the RL control
        if self.config.use_turbulence_control:
            acc += action
            #acc -= self.wind_acceleration(v_wr)
            #acc += np.random.normal(loc=0.0, scale=4.0, size=acc.shape)

        # simulate one time steps of the robots dynamics
        #self._simulate(acc)

        # simulate one time steps of the robots dynamics
        self.p_prev, self.vel_prev = np.copy(self.p), np.copy(self.vel)
        self.p, self.vel = self._simulate_second_order(self.p_prev, self.vel_prev, self.dt, acc, v_wr)

        # Simulate dynamics with only the PD control and assuming no wind
        p_pred, vel_pred = self._simulate_second_order(self.p_prev, self.vel_prev, self.dt, pd_acc)

        #S_error = self.compute_state_error(self.p, self.vel, p_pred, vel_pred)
        #S_error = self.vel - vel_pred
        S_error = np.concatenate([self.p - p_pred, self.vel - vel_pred], axis=-1)

        reward = 0.0
        if self.config.RL_parameters.use_error_mag_reward:
            #reward += -0.1 * (S_error**2).mean()
            wg = self.config.RL_parameters.error_mag_reward_weight
            #reward += - wg * (np.linalg.norm(S_error, axis=-1)).mean()
            control_error = ((self.config.reward_scale * S_error) ** 2).sum(axis=-1)
            reward += - wg * control_error

        # Cosine similarity reward
        if self.config.RL_parameters.use_cosine_reward:
            wg = self.config.RL_parameters.cosine_reward_weight
            cos_error = ((normalize(self.vel, axis=-1) * normalize(vel_pred, axis=-1)).sum(axis=-1) - 1.0)
            reward += wg * cos_error

        # Minimum energy reward
        action_mag = (action ** 2).sum(axis=-1)
        if self.config.RL_parameters.use_action_mag_reward:
            wg = self.config.RL_parameters.action_mag_reward_weight
            #reward += - wg * np.linalg.norm(action, axis=-1).mean()
            reward += - wg * action_mag

        # Computes the formation error
        self.formation_error[:, self.iter] = self._compute_formation_error()
        self.position_error[:, self.iter] = self._compute_position_error()

        if self.config.use_wind_pressure_sensors:

            # gets the wind measurement after applying the action
            new_v_we = self.get_disturbance(self.p)
            # Velocity of the wind with respect to the robot
            new_v_wr = new_v_we - self.vel

            P_d = self._get_pressure(new_v_wr)

            # Observations are a flatten version of the state matrix
            if self.config.measurement_noise:
                v_wr_sensor = self._real_wind_to_sensor(new_v_wr)
            else:
                v_wr_sensor = new_v_wr

            v_wr_mag = (v_wr_sensor ** 2).sum(axis=-1, keepdims=True)
            observation = np.concatenate([self.p, self.vel, v_wr_sensor, P_d, v_wr_mag, S_error], axis=1).reshape(-1)
            # observation = self.p.reshape(-1)
        else:
            observation = np.concatenate([self.p, self.vel, S_error], axis=1).reshape(-1)

        # check if any robot is out of the working space. If it is the case creates a big negative reward
        bounds_cond = (self.p[:, 0] < self.bounds[0]) | (self.p[:, 0] > self.bounds[1]) | (
                    self.p[:, 1] < self.bounds[2]) | (self.p[:, 1] > self.bounds[3])

        self.iter += 1
        if np.any(bounds_cond):

            reward[bounds_cond] += -10.0

        if self.iter >= self._max_episode_steps:
            done = True
        else:
            done = False

        metrics_dict = {
            'control_error': control_error,
            'action_mag': action_mag,
        }

        return observation, reward, done, metrics_dict

    def _simulate_second_order(self, p, vel, dt, action, v_wr=None):

        # Compute the wind's drag acceleration
        if v_wr is not None:
            wind_acceleration = self.wind_acceleration(v_wr)
        else:
            wind_acceleration = np.zeros_like(vel)

        p_next = np.copy(p)
        vel_next = np.copy(vel)
        #n = 10
        #for i in range(n):
        p_next += dt * vel_next
        vel_next += dt * (action + wind_acceleration) # TODO: compute the wind_acceleration at t + i*dt/n

        return p_next, vel_next

    def _simulate(self, action):
        # TODO: Simplified to first order dynamics!!!
        velocity_disturbance = self.get_disturbance(self.p)
        for i in range(self.n_agents):
            self.vel[i, :] = action[i, :] + velocity_disturbance[i, :]
            self.p[i, :] += self.dt * self.vel[i, :]

    def wind_acceleration(self, v_wr):
        v_wr_norm = np.linalg.norm(v_wr, axis=1, keepdims=True)
        wind_acc = (0.5 * self.rho * self.c_d * self.area * v_wr_norm / self.m) * v_wr

        return wind_acc

    def get_disturbance(self, query_points):

        if self.config.turbulence_model == 'random':
            self.setup_random_turbulence()

            # D: num_turb_seeds x num_query_points x points_dimension
            D = (self.turbulence_seeds[:, None, :] - query_points[None, ...])
            D_norm = np.linalg.norm(D, axis=2)
            range = self.bounds[1] - self.bounds[0]

            # Computes the velocity disturbance v. For every turbulence_seed is the source of a vector field rotating
            # clockwise (direction=1) or counter clock-wise (direction=-1). The "wind" decreases linearly as as the query
            # point gets away from the source. The total velocity disturbance at a query point is the sum over all the
            # velocity disturbances for all seeds.
            v = range * np.stack([
                self.direction[:, None] * (- D[..., 1] / (D_norm ** 2)),
                self.direction[:, None] * (  D[..., 0] / (D_norm ** 2))
            ], axis=2).sum(axis=0)
        elif self.config.turbulence_model == 'NS':

            device = torch.device('cuda')

            current_t = self.iter * self.dt + self.sim_time_offset
            res_x, res_y = self.wind_sim_dict[self.selected_sim]['v'].shape[1:3]
            all_t = self.wind_sim_dict[self.selected_sim]['t']

            assert current_t <= all_t[
                -1], f"Current simulation time t={current_t} is greater than max. simulation time t_max={all_t[-1]}"

            idx = np.where(all_t >= current_t)[0]
            assert idx.size > 0, f'Theres is no turbulence data for time t={current_t}'

            idx_left = max(idx[0] - 1, 0)
            idx_right = idx_left + 1

            Z = torch.tensor(self.wind_sim_dict[self.selected_sim]['v'][idx_left:idx_right + 1, ...], device=device)
            scaled_query_points = 2*torch.tensor((query_points - self.bounds[0]) / (self.bounds[1] - self.bounds[0]), device=device) - 1
            padded_current_time = 2*((current_t * torch.ones((query_points.shape[0], 1), device=device) - all_t[idx_left]) / (all_t[idx_right] - all_t[idx_left])) - 1.0

            v = grid_sample(
                Z[None, ...].permute(0, 4, 1, 2, 3),
                # This way we make grid sample match the xy format of the points
                torch.cat([scaled_query_points, padded_current_time], dim=1)[None, None, None, ...],
                mode='bilinear',
                padding_mode='border',
                align_corners=True,
            ).view(2, -1).cpu().T.numpy()
        elif self.config.turbulence_model == 'constant':
            rot_mat = np.array([[np.cos(self.wind_theta), -np.sin(self.wind_theta)], [np.sin(self.wind_theta), np.cos(self.wind_theta)]])
            base_wind = np.array([[8.0, 0.0]]) @ rot_mat.T
            v = base_wind * np.ones_like(query_points) # + np.random.normal(loc=0.0, scale=0.5, size=query_points.shape)
        elif self.config.turbulence_model == 'circular':
            theta = np.arctan2(query_points[:, 1], query_points[:, 0])
            v = 10 * np.concatenate([-np.sin(theta)[:, None], np.cos(theta)[:, None]], axis=-1)

        else:
            v = np.zeros_like(self.p)
            #raise ValueError(f'Invalid turbulence model {self.config.turbulence_model}. Try [NS, random, constant].')

        return v

    def compute_formation_reward(self):

        reward = 0
        for i in range(self.n_agents):
            for j in self.G.neighbors(i):
                reward += np.linalg.norm(self.p[i, :] - self.p[j, :]) - \
                          np.linalg.norm(self.formation_ref[i, :] - self.formation_ref[j, :])

        # TODO: Discuss if this operation, I think this error can be +/- and we need to maximaze the reward
        # that sign oscillation can be problematic
        return -np.abs(reward)

    def _compute_formation_error(self):

        error = np.zeros(self.n_agents)
        for i in range(self.n_agents):
            for j in self.G.neighbors(i):
                error[i] += np.linalg.norm(self.p[i, :] - self.p[j, :]) - \
                          np.linalg.norm(self.formation_ref[i, :] - self.formation_ref[j, :])
        return error

    def _compute_position_error(self):
        error = np.linalg.norm(self.leader_goal[None] + (self.formation_ref - self.formation_ref[0:1, :]) - self.p, axis=1)
        return error

    def compute_state_error(self, p_r, vel_r, p_p, vel_p):
        # Robot state in the real world
        #S_r = np.concatenate([p_r, vel_r], axis=1)
        S_r = vel_r
        # Robot state assuming no wind
        #S_p = np.concatenate([p_p, vel_p], axis=1)
        S_p = vel_p

        S_error = S_r - S_p
        return S_error

    def get_render_args(self):

        args_dict = {}

        # Computes the disturbance vector field on a regular grid, for visualization
        if self.config.turbulence_model is not None:
            vf_res = 30
            x, y = np.meshgrid(
                np.linspace(self.bounds[0], self.bounds[1], vf_res),
                np.linspace(self.bounds[2], self.bounds[3], vf_res)
            )
            x = x.reshape(-1)
            y = y.reshape(-1)
            v = self.get_disturbance(np.stack([x, y], axis=1))
            args_dict['wind_field_x_coord'] = x.copy()
            args_dict['wind_field_y_coord'] = y.copy()
            args_dict['wind_field'] = v.copy()
        else:
            v = None

        args_dict['use_turbulence_model'] = self.config.turbulence_model

        # Computes the plot title text
        title = ''
        for i in range(self.last_action.shape[0]):
            sp_x = ' ' if self.last_action[i, 0] >= 0 else ''
            sp_y = ' ' if self.last_action[i, 1] >= 0 else ''
            title = f'{title} | Act. Node {i}=[{sp_x}{self.last_action[i, 0]:.2f}, {sp_y}{self.last_action[i, 1]:.2f}]'
        args_dict['title'] = title

        args_dict['fig_aspect_ratio'] = 18.0 / 9.0  # Aspect ratio of video.
        args_dict['fig_pixel_height'] = self.config.frame_height
        args_dict['dpi'] = 150
        args_dict['bounds'] = self.bounds.copy()
        args_dict['position'] = self.p.copy()
        args_dict['leader_goal'] = self.leader_goal.copy()

        return args_dict

    @staticmethod
    def render_async(args):
        # Computes the disturbance vector field on a regular grid, for visualization
        if args['use_turbulence_model'] is not None:
            x = args['wind_field_x_coord']
            y = args['wind_field_y_coord']
            v = args['wind_field']

        # Computes the plot title text
        title = args['title']

        # Figure aspect ratio.
        fig_aspect_ratio = args['fig_aspect_ratio']  # Aspect ratio of video.
        fig_pixel_height = args['fig_pixel_height']  # 540  # Height of video in pixels.
        dpi = args['dpi']  # Pixels per inch (affects fonts and apparent size of inch-scale objects).

        # Set the figure to obtain aspect ratio and pixel size.
        fig_w = fig_pixel_height / dpi * fig_aspect_ratio  # inches
        fig_h = fig_pixel_height / dpi  # inches
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), constrained_layout=True, dpi=dpi)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        bounds = args['bounds']

        data_xlim = [bounds[0] - 0.5, bounds[1] + 0.5]
        data_ylim = [bounds[2] - 0.5, bounds[3] + 0.5]

        # Set axes limits which display the workspace nicely.
        ax.set_xlim(data_xlim[0], data_xlim[1])
        ax.set_ylim(data_ylim[0], data_ylim[1])

        # Setting axis equal should be redundant given figure size and limits,
        # but gives a somewhat better interactive resizing behavior.
        ax.set_aspect('equal')

        # Draw robots
        p = args['position']
        robot_handle = ax.scatter(p[:, 0], p[:, 1], 10, 'black')

        leader_goal = args['leader_goal']
        goal_handle = ax.scatter(leader_goal[0], leader_goal[1], 10, 'red')

        ax.set_title(title, fontsize=6)

        # Draw the disturbance vector field
        # NOTE: For what ever the reason the bigger the `scale` the smaller the arrows are
        if args['use_turbulence_model'] is not None:
            vf_handle = ax.quiver(x, y, v[:, 0], v[:, 1], color=[0.4, 0.83, 0.97, 0.85], scale=200)

        # plots the action arrows on top of the robots
        # self.action_arrow_handle = self.ax.quiver(self.p[:, 0], self.p[:, 1], self.last_action[:, 0], self.last_action[:, 1], color='orange', scale=200)

        fig.canvas.draw()
        fig.canvas.flush_events()

        # TODO: Add beautiful renderings for wind!

        #self.fig.canvas.draw()
        #self.fig.canvas.flush_events()

        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)

        return image_from_plot

    def render(self, mode='rgb_array'):

        # Computes the disturbance vector field on a regular grid, for visualization
        if self.config.turbulence_model is not None:
            vf_res = 30
            x, y = np.meshgrid(
                np.linspace(self.bounds[0], self.bounds[1], vf_res),
                np.linspace(self.bounds[2], self.bounds[3], vf_res)
            )
            x = x.reshape(-1)
            y = y.reshape(-1)

            v = self.get_disturbance(np.stack([x, y], axis=1))

        # Computes the plot title text
        title = ''
        for i in range(self.last_action.shape[0]):
            sp_x = ' ' if self.last_action[i, 0] >= 0 else ''
            sp_y = ' ' if self.last_action[i, 1] >= 0 else ''
            title = f'{title} | Act. Node {i}=[{sp_x}{self.last_action[i, 0]:.2f}, {sp_y}{self.last_action[i, 1]:.2f}]'

        if self.fig is None:
            plt.ion()

            # Figure aspect ratio.
            fig_aspect_ratio = 18.0 / 9.0  # Aspect ratio of video.
            fig_pixel_height = self.config.frame_height # 540  # Height of video in pixels.
            dpi = 150  # Pixels per inch (affects fonts and apparent size of inch-scale objects).

            # Set the figure to obtain aspect ratio and pixel size.
            fig_w = fig_pixel_height / dpi * fig_aspect_ratio  # inches
            fig_h = fig_pixel_height / dpi  # inches
            self.fig, self.ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), constrained_layout=True, dpi=dpi)
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
            self.robot_handle = self.ax.scatter(self.p[:, 0], self.p[:, 1], 10, 'black')

            self.goal_handle = self.ax.scatter(self.leader_goal[0], self.leader_goal[1], 10, 'red')

            self.ax.set_title(title, fontsize=6)

            # Draw the disturbance vector field
            # NOTE: For what ever the reason the bigger the `scale` the smaller the arrows are
            if self.config.turbulence_model is not None:
                self.vf_handle = self.ax.quiver(x, y, v[:, 0], v[:, 1], color=[0.4, 0.83, 0.97, 0.85], scale=200)

            # plots the action arrows on top of the robots
            #self.action_arrow_handle = self.ax.quiver(self.p[:, 0], self.p[:, 1], self.last_action[:, 0], self.last_action[:, 1], color='orange', scale=200)

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

            # TODO: Add beautiful renderings for wind!

        else:

            self.robot_handle.set_offsets(self.p)

            self.goal_handle.set_offsets(self.leader_goal)

            self.ax.set_title(title, fontsize=8)

            if self.config.turbulence_model is not None:
                self.vf_handle.set_UVC(v[:, 0], v[:, 1])

            '''for i in range(self.last_action.shape[0]):
                #self.action_arrow_handle[i].clear()
                #self.action_arrow_handle.append(self.ax.quiver(self.p[i, 0], self.p[i, 1], self.last_action[i, 0], self.last_action[i, 1], color='orange', scale=200))

                self.action_arrow_handle.remove()
                #self.action_arrow_handle.XY = self.p.copy()
                #self.action_arrow_handle.set_UVC(self.last_action[:, 0], self.last_action[:, 1])
                self.action_arrow_handle = self.ax.quiver(self.p[:, 0], self.p[:, 1], self.last_action[:, 0], self.last_action[:, 1], color='orange', scale=200)'''

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        if mode == 'rgb_array':
            image_from_plot = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_from_plot = image_from_plot.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

            return image_from_plot

    def set_formation_conf(self):
        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.n_agents))
        if self.config.formation_params.formation_type == 0:  # small triangle
            self.formation_ref = np.array([[0.0, 0.0], [1.0, 0.0], [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)]])
            # specify the formation graph
            self.G.add_edges_from([(0, 1), (1, 2), (2, 0)])
        elif self.config.formation_params.formation_type == 1:    # triangle
            self.G.add_edges_from(
                [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (6, 7), (7, 8),
                 (8, 9), (9, 10), (11, 6), (12, 7), (13, 8), (14, 9), (11, 12), (12, 13), (13, 14), (15, 11), (16, 12),
                 (17, 13), (15, 16), (16, 17), (18, 15), (19, 16), (18, 19), (20, 18)])
            self.formation_ref = np.array(
                [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4], [0.0, 5.0], [1, 0], [1, 1], [1, 2], [1, 3],
                 [1, 4], [2, 0], [2, 1], [2, 2], [2, 3], [3, 0], [3, 1], [3, 2], [4, 0], [4, 1], [5, 0]]) * 1.5
        elif self.config.formation_params.formation_type == 2:  # grid 2x2
            pass
        elif self.config.formation_params.formation_type == 3:  # grid 3x3
            pass
        elif self.config.formation_params.formation_type == 4:  # grid 4x4
            pass
        elif self.config.formation_params.formation_type == 5:  # grid 5x5
            pass
        elif self.config.formation_params.formation_type == 6:    # platoon
            self.G.add_edges_from(
                [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12),
                 (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19), (19, 20)])
            self.formation_ref = np.array(
                [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [1.5, 1.5], [2, 2], [2.5, 2.5], [3, 3], [3.5, 3.5], [4, 4],
                 [4.5, 4.5], [5, 5], [5.5, 5.5], [6, 6], [6.5, 6.5], [7, 7], [7.5, 7.5], [8, 8], [8.5, 8.5], [9, 9],
                 [9.5, 9.5], [10, 10]])
        else:
            raise ValueError(f'Invalid formation type {self.config.formation_params.formation_type}. Tyr [0-> "small triangle, "1 -> "Triangle", 2 -> "Platoon"].')

    def plot_episode(self, results_folder, step=0):
        # print(self.error[0:20,0:3],'  ',p)
        plt.title('')
        fig = plt.figure(1)
        ax = fig.add_subplot(1, 1, 1)

        x = self.dt * np.arange(self._max_episode_steps)

        for i in range(0, self.n_agents):
            ax.plot(x[:self.iter], self.formation_error[i, :self.iter], label=f'Robot {i}')

        ax.legend(loc='upper right', prop={'size': 6})
        ax.grid()
        ax.set_ylim(0.6 * self.bounds[0], 0.6 * self.bounds[1])
        ax.set_title(f'Formation error')
        fig.savefig(os.path.join(results_folder, f'step_{step}_formation_error.png'))
        plt.close(fig)
        '''plt.figure(2)

        for i in range(0, self.n_agents):
            plt.plot(range(50, self._max_episode_steps), self.error[50:self._max_episode_steps, i], label='Robot ' + str(i))
        plt.legend(loc='upper right', prop={'size': 6})
        plt.grid()
        plt.savefig('saved_figure_50_MaxSteps.png')
        plt.clf()'''

    def plot_episode_evaluation(self, data_dict, results_folder, step=0):

        # values in the x-axis (time)
        x = self.dt * np.arange(self._max_episode_steps)

        if 'formation_error' in data_dict:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

            #colors = cm.rainbow(np.linspace(0, 1, eval_data.shape[0]))
            colors = cm.Set1(np.linspace(0, 1, data_dict['formation_error'].shape[1]))
            for i in range(0, self.n_agents):
                mean_data = data_dict['formation_error'].mean(axis=0)
                sdt_data = data_dict['formation_error'].std(axis=0)

                ax.plot(x, mean_data[i, :], label=f'Robot {i}', color=colors[i])
                ax.fill_between(x, mean_data[i] - sdt_data[i], mean_data[i] + sdt_data[i], color=colors[i], alpha=0.1)

            ax.legend(loc='upper right', prop={'size': 6})
            ax.grid()
            ax.set_ylim(0.6 * self.bounds[0], 0.6 * self.bounds[1])
            ax.set_title(f'Formation error')
            fig.savefig(os.path.join(results_folder, f'step_{step}_formation_error.png'))
            plt.close(fig)

        # Position error figure
        if 'position_error' in data_dict:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

            # colors = cm.rainbow(np.linspace(0, 1, eval_data.shape[0]))
            colors = cm.Set1(np.linspace(0, 1, data_dict['position_error'].shape[1]))
            for i in range(0, self.n_agents):
                mean_data = data_dict['position_error'].mean(axis=0)
                sdt_data = data_dict['position_error'].std(axis=0)

                ax.plot(x, mean_data[i, :], label=f'Robot {i}', color=colors[i])
                ax.fill_between(x, mean_data[i] - sdt_data[i], mean_data[i] + sdt_data[i], color=colors[i], alpha=0.1)

            ax.legend(loc='upper right', prop={'size': 6})
            ax.grid()
            ax.set_ylim(0.05 * self.bounds[0], 0.4 * self.bounds[1])
            ax.set_title(f'Position error')
            fig.savefig(os.path.join(results_folder, f'step_{step}_position_error.png'))
            plt.close(fig)

            # robot error figure
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

            ax.boxplot(np.transpose(data_dict['position_error'], axes=[0, 2, 1]).reshape(-1, self.n_agents))

            ax.legend(loc='upper right', prop={'size': 6})
            ax.grid()
            ax.set_ylim(0.2 * self.bounds[0], 0.6 * self.bounds[1])
            ax.set_title(f'Position error per robot')
            fig.savefig(os.path.join(results_folder, f'step_{step}_position_error_per_robot.png'))
            plt.close(fig)

    def get_errors(self):

        errors = {
            'formation_error': self.formation_error.copy(),
            'position_error': self.position_error.copy(),
        }

        return errors
