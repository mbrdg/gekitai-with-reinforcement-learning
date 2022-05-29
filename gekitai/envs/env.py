import itertools

import gym
import numpy as np
from gym import spaces

from . import logic


class GekitaiEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 5}

    def __init__(self, render_mode=None, size=4, max_markers=6, win_condition=3):
        assert render_mode in self.metadata['render_modes'] or render_mode is None

        self.action_space = spaces.Dict({'x': spaces.Discrete(size), 'y': spaces.Discrete(size)})
        self.observation_space = spaces.Box(low=0, high=2, shape=(size**2,), dtype=np.uint8)

        self.board = np.zeros((size, size), dtype=np.uint8)
        self.player_switch = itertools.cycle(range(1, 3))
        self.player = next(self.player_switch)

        self.size, self.max_markers, self.win_condition = size, max_markers, win_condition

        if render_mode == 'human':
            import pygame

            pygame.init()
            pygame.display.init()

            self.window_size = 512
            self.window = pygame.display.set_mode(size=(self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

    def reset(self, *, seed=None, return_info=False, options=None):
        super().reset(seed=seed)

        self.board = np.zeros((self.size, self.size), dtype=np.uint8)
        self.player_switch = itertools.cycle(range(1, 3))
        self.player = next(self.player_switch)

        return self.board.flatten()

    def step(self, action):

        # Do not run move() if the space is not empty, returns an invalid action reward
        if self.board[action['x'], action['y']] != 0:
            return self.board, -100, False, {'invalid': 'Tried to play in a occupied space'}

        self.board = logic.move(self.board, self.player, np.array([action['x'], action['y']]))
        self.player = next(self.player_switch)

        config = {'size': self.size, 'markers': self.max_markers, 'win': self.win_condition}

        done, info = logic.is_over(self.board, config)
        reward = logic.reward(self.board, config)
        info['reward'] = reward

        return self.board.flatten(), reward, done, info

    def render(self, mode='human'):
        import pygame

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        pix_square_size = self.window_size / self.size
        for i, line in enumerate(self.board):
            for j, slot in enumerate(line):
                if not slot:
                    continue

                color = (0, 0, 255) if slot == 1 else (255, 0, 0)
                pygame.draw.circle(canvas, color, (np.array([i, j]) + 0.5) * pix_square_size, pix_square_size / 3)

        for x in range(self.size + 1):
            pygame.draw.line(canvas, 0, (0, pix_square_size * x), (self.window_size, pix_square_size * x), width=3)
            pygame.draw.line(canvas, 0, (pix_square_size * x, 0), (pix_square_size * x, self.window_size), width=3)

        if mode == 'human':
            assert self.window is not None

            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata['render_fps'])

        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
