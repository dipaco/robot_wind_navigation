import gym
from gym.spaces import Dict, Discrete, Box, Tuple
import numpy as np


class DummyEnv(gym.Env):
    def __init__(self, config={}):
        self.config = config
        self.action_space = Box(-1, 1, (2,))
        self.observation_space = Box(-1, 1, (2, 2))
        self.p_done = config.get("p_done", 0.1)

    def reset(self):
        return self.observation_space.sample()

    def step(self, action):

        import pdb
        pdb.set_trace()

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