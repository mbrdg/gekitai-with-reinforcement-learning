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


def move(board, player, pos):
    """ Executes a move and returns the changed game board"""

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


def actions(board, *, shuffle=False):
    """Returns all the possible action for a given game board"""

    acts = np.argwhere(board == 0)
    if shuffle:
        rng = np.random.default_rng()
        rng.shuffle(acts)

    return acts


kernels = {'V': np.ones((1, 3), dtype=np.uint8),
           'H': np.ones((3, 1), dtype=np.uint8),
           'UD': np.eye(3, dtype=np.uint8),
           'LD': np.fliplr(np.eye(3, dtype=np.uint8))}


def is_over(board):
    """Determines if the current game state is over or not"""

    if np.count_nonzero(board == 1) == 8:
        return True, {'winner': 1, 'reason': 'All markers placed'}
    if np.count_nonzero(board == 2) == 8:
        return True, {'winner': 2, 'reason': 'All markers placed'}

    for kernel in kernels.values():
        if (convolve2d(board == 1, kernel, 'valid') == 3).any():
            return True, {'winner': 1, 'reason': '3 adjacent markers'}
        if (convolve2d(board == 2, kernel, 'valid') == 3).any():
            return True, {'winner': 2, 'reason': '3 adjacent markers'}

    return False, {'winner': 0}


def weights(shape):
    """Builds a weight matrix for the game board"""

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
