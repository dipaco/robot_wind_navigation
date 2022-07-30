import os
import sys
import numpy as np

__BASE_FOLDER__ = os.path.dirname(os.path.abspath(__file__))
dist_package_folder = os.path.join(__BASE_FOLDER__, 'submodules/pytorch_sac/')
sys.path.append(dist_package_folder)

from replay_buffer import ReplayBuffer


class MultiRobotReplayBuffer(ReplayBuffer):
    def __init__(self, obs_shape, action_shape, capacity, num_robots, device):
        super().__init__(obs_shape, action_shape, capacity, device)
        self.num_robots = num_robots
        self.rewards = np.empty((capacity, self.num_robots), dtype=np.float32)
