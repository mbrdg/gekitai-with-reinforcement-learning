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
    """ Executes a move and returns the changed board.

    Parameters
    ----------
    board : ndarray
        Current board according to the state of the game
    player : np.uint8
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


def actions(board, *, shuffle=False):
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
        return rng.shuffle(acts)

    return acts


kernels = {'V': np.ones((1, 3), dtype=np.uint8),
           'H': np.ones((3, 1), dtype=np.uint8),
           'UD': np.eye(3, dtype=np.uint8),
           'LD': np.fliplr(np.eye(3, dtype=np.uint8))}


def reward(board, config, *, invert=False, threat_weight=1.0, marker_weight=1.0):
    """Calculates the reward given a board.
    In this case positive scores means that Player 1 has an advantage over its opponent.
    Nevertheless, it is possible to invert the reward value by passing invert=True.

    Parameters
    ----------
    board : ndarray
        Current board according to the state of the game
    config : dict
        Set of parameters that describe some game rules
    invert : bool
        If True, then a positive score means that Player 2 has an advantage over Player 1
    threat_weight : float
        Weight for a threat
    marker_weight : float
        Weight for a marker

    Returns
    -------
    flaat
        Reward value for the current board state
    """

    def threat_calc(p):
        return sum((convolve2d(board == p, kernel, mode='valid') == config['win'] - 1).any() for kernel in kernels)

    threats = (threat_calc(1) - threat_calc(2)) * threat_weight
    placed = (np.count_nonzero(board == 1) - np.count_nonzero(board == 2)) * marker_weight

    return -(placed + threats) if invert else (placed + threats)


def is_over(board, config):
    """Determines if the current game state is over or not

    Parameters
    ----------
    board : ndarray
        Current board according to the state of the game
    config : dict
        Set of parameters that describe some game rules

    Returns
    -------
    bool
        A boolean that tells if the game is over or not
    info
        A description of the outcome of detecting if the game is over
    """

    # Determine if a player already placed its markers
    if np.count_nonzero(board == 1) >= config['markers']:
        return True, {'over': 'Player 1 won by placing all of their markers'}
    if np.count_nonzero(board == 2) >= config['markers']:
        return True, {'over': 'Player 2 won by placing all of their markers'}

    # Check if a player has 3 adjacent markers
    for kernel in kernels.values():
        if (convolve2d(board == 1, kernel, mode='valid') == 3).any():
            return True, {'over': 'Player 1 won by placing at least 3 adjacent markers'}
        if (convolve2d(board == 2, kernel, mode='valid') == 3).any():
            return True, {'over': 'Player 2 won by placing at least 3 adjacent markers'}

    return False, {'over': 'The game is not yet over'}
