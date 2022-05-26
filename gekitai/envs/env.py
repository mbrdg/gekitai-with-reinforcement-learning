import itertools

import numpy as np
import gym
from gym import spaces

from . import logic


class GekitaiEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 5}

    def __init__(self, render_mode=None, size=4):
        assert render_mode in self.metadata['render_modes'] or render_mode is None

        self.action_space = spaces.Discrete(size * size)
        self.observation_space = spaces.Box(low=0, high=2, shape=(size, size), dtype=np.uint8)

        self.board_size = size
        self.board = np.zeros((size, size), dtype=np.uint8)
        self.player_switch = itertools.cycle(range(1, 3))
        self.player = next(self.player_switch)

    def reset(self, *, seed=None, return_info=False, options=None):
        super().reset(seed=seed)

        self.board = np.zeros((self.board_size, self.board_size), dtype=np.uint8)
        info = dict()

        return (self.board, info) if return_info else self.board_size

    # TODO: calculate, reward and improve information
    def step(self, action):
        moves = logic.actions(self.board)
        action = moves[action % len(moves)]

        self.board = logic.move(self.board, self.player, action)
        self.player = next(self.player_switch)

        done = logic.is_over(self.board)
        reward = 0.0
        info = dict()

        return self.board, reward, done, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass
