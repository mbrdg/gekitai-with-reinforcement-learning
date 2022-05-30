from typing import Tuple
from copy import deepcopy

import numpy as np
from scipy.signal import convolve2d

# Masks for pushing the pieces
masks = {'N': np.array([[-1, 0], [-2, 0]], dtype=np.int8),
         'NE': np.array([[-1, 1], [-2, 2]], dtype=np.int8),
         'E': np.array([[0, 1], [0, 2]], dtype=np.int8),
         'SE': np.array([[1, 1], [2, 2]], dtype=np.int8),
         'S': np.array([[1, 0], [2, 0]], dtype=np.int8),
         'SW': np.array([[1, -1], [2, -2]], dtype=np.int8),
         'W': np.array([[0, -1], [0, -2]], dtype=np.int8),
         'NW': np.array([[-1, -1], [-2, -2]], dtype=np.int8)}


def move(board: np.ndarray, player: int, pos: np.ndarray) -> np.ndarray:
    """ Executes a move and returns the changed board.

    Parameters
    ----------
    board : ndarray
        Current board according to the state of the game
    player : int
        Player identifier in which the move will be executed
    pos : ndarray
        Position where player will place its piece

    Returns
    -------
    ndarray
        The board with the expected changes, i.e. the new game state.

    Raises
    ------
    AssertionError
        If the shape of pos does not match (2,) or if the provided position
        is not empty.
    """

    assert board.shape[0] == board.shape[1], pos.shape == (2,)
    i, j = pos[0], pos[1]
    size = board.shape[0]

    assert board[i, j] == 0
    board[i, j] = player

    for mask in masks.values():
        ngh, opp = pos + mask[0], pos + mask[1]
        x0, y0, x1, y1 = ngh[0], ngh[1], opp[0], opp[1]

        # Out of bounds
        if np.any(np.isin(ngh, range(size), invert=True)):
            continue

        # Nothing to push, just proceed
        if board[x0, y0] == 0:
            continue

        # Push the piece outside the board
        if np.any(np.isin(opp, range(size), invert=True)):
            board[x0, y0] = 0
            continue

        # It is possible to push the piece, swap the values
        if board[x1, y1] == 0:
            board[x1, y1], board[x0, y0] = board[x0, y0], 0

    return board


def actions(board: np.ndarray, *, shuffle: bool = False) -> np.ndarray:
    """Retrieves all the possible actions from a board

    Parameters
    ----------
    board : ndarray
        Current board according to the state of the game
    shuffle : bool
        Shuffles the provided array of possible actions

    Returns
    -------
    ndarray
        An array containing all the possible actions
    """

    acts = np.argwhere(board == 0)

    if shuffle:
        rng = np.random.default_rng()
        rng.shuffle(acts)

    return acts


kernels = {'V': np.ones((1, 3), dtype=np.uint8),
           'H': np.ones((3, 1), dtype=np.uint8),
           'UD': np.eye(3, dtype=np.uint8),
           'LD': np.fliplr(np.eye(3, dtype=np.uint8))}


def is_over(board: np.ndarray):
    """Determines if the current game state is over or not

    Parameters
    ----------
    board : ndarray
        Current board according to the state of the game

    Returns
    -------
    info
        A description of the outcome of detecting if the game is over
    """

    # Determine if a player already placed its markers
    if np.count_nonzero(board == 1) == 8:
        return True, {'winner': 1, 'reason': 'All markers placed'}
    if np.count_nonzero(board == 2) == 8:
        return True, {'winner': 2, 'reason': 'All markers placed'}

    # Check if a player has 3 adjacent markers
    for kernel in kernels.values():
        if (convolve2d(board == 1, kernel, 'valid') == 3).any():
            return True, {'winner': 1, 'reason': '3 adjacent markers'}
        if (convolve2d(board == 2, kernel, 'valid') == 3).any():
            return True, {'winner': 2, 'reason': '3 adjacent markers'}

    return False, {'winner': 0}


def weights(shape):
    assert shape[0] == shape[1]

    if shape[0] % 2:
        init_value, init_shape = shape[0] // 2 + 1, (1, 1)
    else:
        init_value, init_shape = (shape[0] // 2), (2, 2)

    weights_ = np.empty(shape=init_shape, dtype=np.uint8)
    weights_.fill(init_value)

    for i in range(init_value - 1, 0, -1):
        weights_ = np.pad(weights_, 1, 'constant', constant_values=i)

    return weights_


def eval_board(board: np.ndarray, config: dict, *, invert: bool = False) -> float:
    """Calculates the reward given a board.
    In this case positive scores means that Player 1 has an advantage over its opponent.
    Nevertheless, it is possible to invert the reward value by passing invert=True.
    Parameters
    ----------
    board : ndarray
        Current board according to the state of the game
    config : dict
        Set of parameters that describes the rules of the game
    invert : bool
        If True, then a positive score means that Player 2 has an advantage over Player 1
    Returns
    -------
    flaat
        Reward value for the current board state
    """
    size, threat_cond = config['size'], config['win_condition'] - 1

    center = size // 2
    space_weights = np.zeros(shape=(size, size))
    for i in range(size):
        for j in range(size):
            space_weights[i, j] = max(abs(center - i), abs(center - j))

    placed = np.sum((board == 1) * space_weights) - np.sum((board == 2) * space_weights)
    threats = sum(
        np.count_nonzero(convolve2d(board == 1, kernel, 'valid') == threat_cond) -
        np.count_nonzero(convolve2d(board == 2, kernel, 'valid') == threat_cond)
        for kernel in kernels.values()
    )

    return placed + threats if not invert else -(placed + threats)
