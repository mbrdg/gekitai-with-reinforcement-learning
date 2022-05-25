from typing import Optional, Union, Tuple

import gym
from gym import spaces
import numpy as np


class GekitaiEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'render_fps': 5}

    def __init__(self, render_mode: Optional[str] = None, size: int = 4):
        assert render_mode is None or render_mode in self.metadata['render_modes']

        self.size = size
        self.window_size = 512

        # Action Space represents the number of possible actions
        # Therefore we use the size of the board to initialize that information
        self.action_space = spaces.Discrete(size * size)

        # Observation Space represents the positions where there is already a marker
        self.observation_space = spaces.Dict({
            'ally': spaces.Box(0, 8, shape=(2,), dtype=np.uint8),
            'rival': spaces.Box(0, 8, shape=(2,), dtype=np.uint8)
        })

        if render_mode == "human":
            import pygame

            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

    def reset(self, *, seed=None, return_info=False, options=None):
        super().reset(seed=seed)
        return

    def step(self, action):
        raise NotImplementedError()

    def render(self, mode='human'):
        return self._render_frame(mode)

    def _render_frame(self, mode: str):
        import pygame

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        if mode == 'human':
            assert self.window is not None

            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata['render_fps'])
        elif mode == 'rgb_array':
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 2))

    def close(self):
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
