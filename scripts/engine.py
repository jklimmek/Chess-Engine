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
    Thus, it is recommended to set the search depth to 3 or lower. Currently model can only play as white.

    # todo: Implement opening book.
    Model was trained to compare positions starting from the 5th move, so it struggles in the opening.
    Thus, during the opening the engine will either mimic the opponent's move or play one of the 
    hardcoded openings.
    """

    def __init__(self, model, color, depth, input_format="uci", device="cpu"):
        """Initializes the engine.

        Args:
            model (torch.nn.Module): The model to use for evaluation.
            color (bool): The color of the player.
            depth (int): The depth of the search tree.
            input_format (str): The input format. Can be either "uci" or "san".
            device (str): The device to use for evaluation. Can be either "cpu" or "cuda".
        """
        
        # Initialize the model.
        self._model = model
        self._model.eval()
        self._model.to(device)
        # todo: Implement posibility for model to play as black.
        self._color = color
        self._depth = depth
        self._input_format = input_format
        self._device = device

        # Constants for alpha-beta pruning.
        self.MIN_INF = float("-inf")
        self.MAX_INF = float("inf")

        # Condition to break out of the game after the player makes types `break` in the console.
        # Without it kernel will crash when interrupting the game.
        self.stop_game = False

        # Hash table to store the bitboard of each position.
        # This speeds up the search significantly.
        self.position_hash_table = {}

        # Number of times the engine retrieved a position from the hash table.
        self.retrieved_from_hash_table = 0


    def __hash_position(self, board):
        """Hashes the current position of the board."""
        return hashlib.md5(board.fen().encode()).hexdigest()


    def __get_cached_values(self, board):
        """Returns the cached value of the current position if it exists."""
        key = self.__hash_position(board)
        if key in self.position_hash_table:
            return self.position_hash_table[key]
        return None


    def __cache_values(self, board, bitboard):
        """Caches the current position and its value."""
        key = self.__hash_position(board)
        self.position_hash_table[key] = bitboard


    def __get_bitboard(self, board):
        """Returns the bitboard of the current position."""
        cached_values = self.__get_cached_values(board)
        if cached_values is not None:
            self.retrieved_from_hash_table += 1
            return cached_values
        bitboard = torch.tensor(fen_to_array(board), dtype=torch.float32)[None, :].to(self._device)
        bitboard = self._model.extract(bitboard)
        self.__cache_values(board, bitboard)
        return bitboard
    

    def __predict(self, board1, board2):
        """Returns the board with the higher value."""
        board1_bitboard = self.__get_bitboard(board1)
        board2_bitboard = self.__get_bitboard(board2)
        prediction = self._model.compare(board1_bitboard, board2_bitboard)[0]
        if prediction[0] > prediction[1]:
            return board1, board2
        return board2, board1
    

    def __minimax_alpha_beta(self, board, depth, alpha, beta, color):
        """Returns the value of the current position.
        
        Uses comparison-based minimax with alpha-beta pruning.
        
        Args:
            board (chess.Board): The current position.
            depth (int): The depth of the search tree.
            alpha (float): The alpha value.
            beta (float): The beta value.
            color (bool): The color of the player.
        """

        # If the game is over, return return board.
        if depth == 0:
            return board

        # White's turn.
        if color:
            v = self.MIN_INF
            for move in board.legal_moves:
                cur = board.copy()
                cur.push(move)
                if v == self.MIN_INF:
                    v = self.__minimax_alpha_beta(cur, depth-1, alpha, beta, False) 
                if alpha == self.MIN_INF:
                    alpha = v
            
                v = self.__predict(v, self.__minimax_alpha_beta(cur, depth-1, alpha, beta, False))[0]
                alpha = self.__predict(alpha, v)[0] 
                if beta != self.MAX_INF:
                    if self.__predict(alpha, beta)[0] == alpha:
                        break
            return v 

        # Black's turn.
        else:
            v = self.MAX_INF
            for move in board.legal_moves:
                cur = board.copy()
                cur.push(move)
                if v == self.MAX_INF:
                    v = self.__minimax_alpha_beta(cur, depth-1, alpha, beta, True) 
                if beta == self.MAX_INF:
                    beta = v
                
                v = self.__predict(v, self.__minimax_alpha_beta(cur, depth-1, alpha, beta, True))[1]
                beta = self.__predict(beta, v)[1] 
                if alpha != self.MIN_INF:
                    if self.__predict(alpha, beta)[0] == alpha:
                        break
            return v 


    def engine_move(self, board):
        """
        Selects the best move for the AI engine using the minimax algorithm with alpha-beta pruning.

        Args:
            board (chess.Board): The current position on the chessboard.

        Returns:
            chess.Board: The updated chessboard after the AI engine makes its move.
        """

        # Initialize alpha and beta values for alpha-beta pruning.
        alpha = self.MIN_INF
        beta = self.MAX_INF

        # Initialize the evaluation value to negative infinity.
        v = self.MIN_INF

        # Iterate through legal moves to find the best move.
        for move in board.legal_moves:
            # Create a copy of the board to simulate the move.
            cur = board.copy()
            cur.push(move)

            # If it's the first move, perform the initial minimax alpha-beta search.
            if v == self.MIN_INF:
                v = self.__minimax_alpha_beta(cur, self._depth-1, alpha, beta, False)
                bestMove = move

                # If alpha is still at its initial value, update it with the current evaluation.
                if alpha == self.MIN_INF:
                    alpha = v

            # For subsequent moves, update the evaluation and alpha value based on predictions.
            else:
                # Predict the new evaluation value based on the minimax search result.
                new_v = self.__predict(self.__minimax_alpha_beta(cur, self._depth-1, alpha, beta, False), v)[0]

                # If the new evaluation is different, update the best move and evaluation.
                if new_v != v:
                    bestMove = move
                    v = new_v

                # Update alpha with the current evaluation.
                alpha = self.__predict(alpha, v)[0]

        # Push the best move to the board and return the updated board.
        board.push(bestMove)
        return board


    def player_move(self, board):
        """
        Allows the player to make a move on the chessboard. Handles user input and updates the board accordingly.

        Args:
            board (chess.Board): The current position on the chessboard.

        Returns:
            chess.Board: The updated chessboard after the player makes a valid move.
        """

        while True:
            try:
                move = input()

                # Check if the player wants to break out of the game.
                if move == "break":
                    self.stop_game = True
                    break

                # Attempt to apply the player's move to the board.
                board.push_uci(move) if self._input_format == "uci" else board.push_san(move)
                break

            # Handle the case where the input is not a valid move.
            except ValueError:
                pass

        # Return the updated board after the player makes a move.
        return board