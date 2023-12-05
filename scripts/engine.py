import hashlib
import torch

from .utils import *


class Engine:
    """A class to represent the engine.
    
    Uses comparison-based minimax with alpha-beta pruning to find the best move.
    In order to speed up the search, the engine caches the bitboard of each position.
    Search speed depends on the depth of the search tree and the number of positions in the cache.
    Usually after a few moves search speed increases significantly.

    One thing to keep in mind is that speed depends on search depth and the number of positions in the cache.
    Thus, it is recommended to set the search depth to 3 or lower.

    Model was trained to compare positions starting from the 5th move, so it struggles in the opening.
    Thus, during the opening the engine will either mimic the opponent's move or play one of the 
    hardcoded openings.
    """


    def __init__(self, model, color, depth, device="cpu"):
        """Initializes the engine.

        Args:
            model (Model): The model to be used by the engine.
            color (bool): The color of the engine.
            depth (int): The depth of the search tree.
            device (str, optional): The device to be used by the engine. Defaults to "cpu".
        """

        self._model = model
        self._model.eval()
        self._model.to(device)
        self._color = color
        self._depth = depth
        self._device = device
        self.move = None
        self.total_moves = 0

        self.MIN_INF = float("-inf")
        self.MAX_INF = float("inf")

        self._position_hash_table = {}


    def __hash_position(self, board):
        """Hashes the current position of the board."""

        return hashlib.md5(board.fen().encode()).hexdigest()


    def __get_cached_values(self, board):
        """Returns the cached value of the current position if it exists."""

        key = self.__hash_position(board)
        if key in self._position_hash_table:
            return self._position_hash_table[key]
        return None


    def __cache_values(self, board, bitboard):
        """Caches the current position and its value."""

        key = self.__hash_position(board)
        self._position_hash_table[key] = bitboard


    def __get_bitboard(self, board):
        """Returns the bitboard of the current position."""

        cached_values = self.__get_cached_values(board)
        if cached_values is not None:
            return cached_values
        bitboard = torch.tensor(fen_to_array(board), dtype=torch.float32)[None, :].to(self._device)
        bitboard = self._model.extract(bitboard)
        self.__cache_values(board, bitboard)
        return bitboard
    

    def __minimax_alpha_beta(self, board, color, depth, alpha, beta):
        """Returns the best move using minimax with alpha-beta pruning.

        Args:
            board (chess.Board): The current board.
            color (bool): The color of the engine.
            depth (int): The depth of the search tree.
            alpha (float): The alpha value.
            beta (float): The beta value.

        Returns:
            float: The value of the current position.
        """

        # If the terminal node is reached or the game is over, return the current position.
        if depth >= self._depth or board.is_game_over():
            return board
        
        # White's turn.
        if color:
            v = self.MIN_INF
            for move in board.legal_moves:
                n_board = board.copy()
                n_board.push(move)

                # First move in the branch.
                if v == self.MIN_INF:
                    v = self.__minimax_alpha_beta(n_board, False, depth+1, alpha, beta)
                    v_bitboard = self.__get_bitboard(v)

                # Subsequent moves, compare and update based on the model evaluation.
                else:
                    v_bitboard = self.__get_bitboard(v)
                    n_v = self.__minimax_alpha_beta(n_board, False, depth+1, alpha, beta)
                    n_v_bitboard = self.__get_bitboard(n_v)
                    best_v = self._model.compare(v_bitboard, n_v_bitboard)[0]
                    if best_v[1] > best_v[0]:
                        v = n_v
                        v_bitboard = n_v_bitboard
                
                # Update alpha if it is the initial value.
                if alpha == self.MIN_INF:
                    alpha = v

                alpha_bitboard = self.__get_bitboard(alpha)
                scores_alpha = self._model.compare(v_bitboard, alpha_bitboard)[0]

                # Update alpha if the new move is better.
                if scores_alpha[0] > scores_alpha[1]:
                    alpha = v

                if beta != self.MAX_INF:
                    beta_bitboard = self.__get_bitboard(beta)
                    scores_beta = self._model.compare(v_bitboard, beta_bitboard)[0]

                    # Prune the branch if beta is less than the max value.
                    if scores_beta[0] > scores_beta[1]:
                        break
            return v
        
        # Black's turn.
        else:
            v = self.MAX_INF
            for move in board.legal_moves:
                n_board = board.copy()
                n_board.push(move)

                # First move in the branch.
                if v == self.MAX_INF:
                    v = self.__minimax_alpha_beta(n_board, True, depth+1, alpha, beta)
                    v_bitboard = self.__get_bitboard(v)

                # Subsequent moves, compare and update based on the model evaluation.
                else:
                    v_bitboard = self.__get_bitboard(v)
                    n_v = self.__minimax_alpha_beta(n_board, True, depth+1, alpha, beta)
                    n_v_bitboard = self.__get_bitboard(n_v)
                    best_v = self._model.compare(v_bitboard, n_v_bitboard)[0]
                    if best_v[0] > best_v[1]:
                        v = n_v
                        v_bitboard = n_v_bitboard

                # Update beta if it is the initial value.
                if beta == self.MAX_INF:
                    beta = v

                v_bitboard = self.__get_bitboard(v)
                beta_bitboard = self.__get_bitboard(beta)
                scores_beta = self._model.compare(beta_bitboard, v_bitboard)[0]

                # Update beta if the new move is better.
                if scores_beta[0] > scores_beta[1]:
                    beta = v

                if alpha != self.MIN_INF:
                    alpha_bitboard = self.__get_bitboard(alpha)
                    scores_alpha = self._model.compare(alpha_bitboard, v_bitboard)[0]

                    # Prune the branch if alpha is greater than the min value.
                    if scores_alpha[0] > scores_alpha[1]:
                        break

            return v


    def play(self, board):
        """Returns the best move for the current position.

        Args:
            board (chess.Board): The current board.

        Returns:
            chess.Move: The best move for the current position.
        """

        v = self.MIN_INF
        alpha = self.MIN_INF
        beta = self.MAX_INF

        for move in board.legal_moves:
            n_board = board.copy()
            n_board.push(move)
            
            # First move in the branch.
            if v == self.MIN_INF:
                v = n_board
                self.move = move

                # Initialize alpha if it is the initial value.
                if alpha == self.MIN_INF:
                    alpha = v
                continue
            
            v_bitboard = self.__get_bitboard(v)

            # todo: currently model can only play as white, add support for black
            n_v = self.__minimax_alpha_beta(n_board, False, 1, alpha, beta)
            n_v_bitboard = self.__get_bitboard(n_v)
            best_v = self._model.compare(v_bitboard, n_v_bitboard)[0]

            # Update v and its bitboard if the new move is better.
            if best_v[1] > best_v[0]:
                v = n_v
                v_bitboard = n_v_bitboard

            alpha_bitboard = self.__get_bitboard(alpha)
            scores_alpha = self._model.compare(v_bitboard, alpha_bitboard)[0]

            # Update alpha and the selected move if the new move is better.
            if scores_alpha[0] > scores_alpha[1]:
                alpha = v
                self.move = move