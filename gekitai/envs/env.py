import itertools

import gym
import numpy as np
from gym import spaces

from .logic import actions, move, is_over, weights


class GekitaiEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 5}

    def __init__(self, render_mode=None, size=6, markers=8):
        assert render_mode in self.metadata['render_modes'] or render_mode is None

        self.board_size, self.max_markers = size, markers

        self.action_space = spaces.Discrete(size ** 2)
        self.observation_space = spaces.Box(low=0, high=2, shape=(size ** 2,), dtype=np.uint8)

        self.board = np.zeros(shape=(size, size), dtype=np.uint8)
        self.player_switch = itertools.cycle(range(1, 3))
        self.player = next(self.player_switch)

        if render_mode == 'human':
            import pygame

            pygame.init()
            pygame.display.init()

            self.window_size = 512
            self.window = pygame.display.set_mode(size=(self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

    def reset(self, *, seed=None, return_info=False, options=None):
        super().reset(seed=seed)

        self.board = np.zeros(shape=(self.board_size, self.board_size), dtype=np.uint8)
        self.player_switch = itertools.cycle(range(1, 3))
        self.player = next(self.player_switch)

        return self.board.flatten()

    def step(self, action):
        i, j = action // self.board_size, action % self.board_size

        if self.board[i, j] != 0:
            return self.board, -10, False, {'desc': 'Invalid action chosen'}

        self.board = move(self.board, self.player, np.array([i, j]))
        self.player = next(self.player_switch)

        done, info = is_over(self.board)
        if done:
            return self.board.flatten(), 1, True, info

        rng = np.random.default_rng()

        board_weights = weights(self.board.shape)
        possible_actions = actions(self.board)

        probs = np.array([board_weights[m[0], m[1]] for m in possible_actions])
        bot_move = rng.choice(possible_actions, p=np.array([m / np.sum(probs) for m in probs]))

        self.board = move(self.board, self.player, bot_move)
        self.player = next(self.player_switch)

        done, info = is_over(self.board)
        if done:
            return self.board.flatten(), -1, True, info

        return self.board.flatten(), 0, False, info

    def render(self, mode='human'):
        import pygame

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        pix_square_size = self.window_size / self.board_size
        for i, line in enumerate(self.board):
            for j, slot in enumerate(line):
                if not slot:
                    continue

                color = (0, 0, 255) if slot == 1 else (255, 0, 0)
                pygame.draw.circle(canvas, color, (np.array([i, j]) + 0.5) * pix_square_size, pix_square_size / 3)

        for x in range(self.board_size + 1):
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
