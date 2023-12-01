import hashlib
import torch

from .utils import *


class Engine:
    def __init__(self, model, color, depth, device="cpu"):
        self._model = model
        self._model.eval()
        self._model.to(device)
        self._color = color
        self._depth = depth
        self._device = device
        self.move = None

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
        if depth > self._depth or board.is_game_over():
            return board
        
        # white's turn
        if color:
            v = self.MIN_INF
            for move in board.legal_moves:
                n_board = board.copy()
                n_board.push(move)

                if v == self.MIN_INF:
                    v = self.__minimax_alpha_beta(n_board, False, depth+1, alpha, beta)
                    v_bitboard = self.__get_bitboard(v)
                else:
                    v_bitboard = self.__get_bitboard(v)
                    n_v = self.__minimax_alpha_beta(n_board, False, depth+1, alpha, beta)
                    n_v_bitboard = self.__get_bitboard(n_v)
                    best_v = self._model.compare(v_bitboard, n_v_bitboard)[0]
                    if best_v[1] > best_v[0]:
                        v = n_v
                        v_bitboard = n_v_bitboard
                
                if alpha == self.MIN_INF:
                    alpha = v

                alpha_bitboard = self.__get_bitboard(alpha)
                scores_alpha = self._model.compare(v_bitboard, alpha_bitboard)[0]

                if scores_alpha[0] > scores_alpha[1]:
                    alpha = v

                if beta != self.MAX_INF:
                    beta_bitboard = self.__get_bitboard(beta)
                    scores_beta = self._model.compare(v_bitboard, beta_bitboard)[0]
                    if scores_beta[0] > scores_beta[1]:
                        break
            return v
        # black's turn
        else:
            v = self.MAX_INF
            for move in board.legal_moves:
                n_board = board.copy()
                n_board.push(move)

                if v == self.MAX_INF:
                    v = self.__minimax_alpha_beta(n_board, True, depth+1, alpha, beta)
                    v_bitboard = self.__get_bitboard(v)
                else:
                    v_bitboard = self.__get_bitboard(v)
                    n_v = self.__minimax_alpha_beta(n_board, True, depth+1, alpha, beta)
                    n_v_bitboard = self.__get_bitboard(n_v)
                    best_v = self._model.compare(v_bitboard, n_v_bitboard)[0]
                    if best_v[0] > best_v[1]:
                        v = n_v
                        v_bitboard = n_v_bitboard


                if beta == self.MAX_INF:
                    beta = v

                v_bitboard = self.__get_bitboard(v)
                beta_bitboard = self.__get_bitboard(beta)

                scores_beta = self._model.compare(beta_bitboard, v_bitboard)[0]

                if scores_beta[0] > scores_beta[1]:
                    beta = v

                if alpha != self.MIN_INF:
                    alpha_bitboard = self.__get_bitboard(alpha)
                    scores_alpha = self._model.compare(alpha_bitboard, v_bitboard)[0]
                    if scores_alpha[0] > scores_alpha[1]:
                        break

            return v


    def play(self, board):

        v = self.MIN_INF
        alpha = self.MIN_INF
        beta = self.MAX_INF

        for move in board.legal_moves:
            n_board = board.copy()
            n_board.push(move)

            if v == self.MIN_INF:
                v = n_board
                self.move = move
                if alpha == self.MIN_INF:
                    alpha = v
                continue
            
            v_bitboard = self.__get_bitboard(v)

            n_v = self.__minimax_alpha_beta(n_board, self._color, 1, alpha, beta)
            n_v_bitboard = self.__get_bitboard(n_v)

            best_v = self._model.compare(v_bitboard, n_v_bitboard)[0]
            if best_v[1] > best_v[0]:
                v = n_v
                v_bitboard = n_v_bitboard

            alpha_bitboard = self.__get_bitboard(alpha)
            scores_alpha = self._model.compare(v_bitboard, alpha_bitboard)[0]

            if scores_alpha[0] > scores_alpha[1]:
                alpha = v
                self.move = move

# Depth 3 == 9 sec
# Depth 4 == 20 sec; mid game 70, 120 sec
# Depth 5 == 49 sec
# Depth 6 == 

# model = DeepChess(AE().encoder)
# _ = load_state_dict("../runs/deepchess_bs512_lr0.01/epoch-38_loss-0.2100_train_acc-0.9053_dev_acc-0.8913.pth", model)
# engine = Engine(model=model, color=True, depth=3, device="cpu")