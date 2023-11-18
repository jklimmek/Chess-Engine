import argparse
import logging
import os
import random

import chess.pgn
from tqdm import tqdm

from .utils import *


def extract_random_positions_from_game(game, num_positions, skip_first_n_moves=5):
    """
    Extracts random positions from a game.

    Args:
        game (chess.pgn.Game): A chess.pgn.Game object.
        num_positions (int): The number of positions to extract.
        skip_first_n_moves (int): The number of moves to skip at the beginning of the game.

    Returns:
        A list of FEN strings.
    """

    # Initialize an empty list to store the positions.
    positions = []

    # Create a new board object from the game.
    board = game.board()

    # Get all the moves in the mainline of the game.
    all_moves = list(game.mainline_moves())

    # Loop through all the moves in the game.
    for index, move in enumerate(all_moves):

        # Skip the first n moves.
        if index >= skip_first_n_moves and not board.is_capture(move):

            # If the move is not a capture, add the current position to the list of positions.
            positions.append(board.fen())

        # Make the move on the board.
        board.push(move)

    # If there are not enough positions, return all the positions.
    if len(positions) < num_positions:
        return positions

    # Otherwise, return a random sample of the positions.
    return random.sample(positions, num_positions)


def process_pgn_file(file_path, num_positions, skip_first_n_moves=5):
    """
    Extracts random positions from a PGN file.

    Args:
        file_path (str): The path to the PGN file.
        num_positions (int): The number of positions to extract.
        skip_first_n_moves (int): The number of moves to skip at the beginning of the game.

    Returns:
        A tuple of two lists of FEN strings. The first list contains positions from games
        where white won, and the second list contains positions from games where black won.
    """

    # Initialize empty lists to store positions.
    white_wins_positions = []
    black_wins_positions = []

    # Initialize counters for number of games where white and black won.
    white_games, black_games = 0, 0

    # Initialize progress bar.
    pbar = tqdm()

    # Open PGN file.
    with open(file_path, "r") as pgn_file:

        # Loop through games in PGN file.
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            # Get result of game.
            result = game.headers["Result"]

            # If white won, extract positions from game and add to white_wins_positions list.
            if result.startswith("1-0"):
                white_wins_positions.extend(extract_random_positions_from_game(game, num_positions, skip_first_n_moves))
                white_games += 1

            # If black won, extract positions from game and add to black_wins_positions list.
            elif result.startswith("0-1"):
                black_wins_positions.extend(extract_random_positions_from_game(game, num_positions, skip_first_n_moves))
                black_games += 1

            # Update progress bar.
            pbar.update(1)

    # Close progress bar.
    pbar.close()

    # Return tuple of positions and number of games won by white and black.
    return white_wins_positions, black_wins_positions, white_games, black_games


def validate_two_file_names(value):
    """Validates that exactly two file names are provided."""

    values = value.split(',')
    if len(values) != 2:
        raise argparse.ArgumentTypeError("Exactly two file names are required.")
    return values


def parse_args():
    """Parses command line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn-file", type=str, required=True, help="The path to the PGN file.")
    parser.add_argument("--output-dir", type=str, required=True, help="The directory to save the positions.")
    parser.add_argument("--num-positions", type=int, default=10, help="The number of positions to extract from each game. If there are fewer positions in a game, all positions will be extracted.")
    parser.add_argument("--skip-first-n-moves", type=int, default=5, help="The number of moves to skip at the beginning of each game.")
    parser.add_argument("--file-names", default="white.txt,black.txt", type=validate_two_file_names, help="Specify exactly two file names separated by a comma. E.g. white.txt,black.txt")
    parser.add_argument("--verbose", action="store_true", help="Whether to print logs.")
    return parser.parse_args()


def main():

    # Parse command line arguments.
    args = parse_args()

    # Process PGN file to extract positions.
    white_pos, black_pos, white_games, black_games = process_pgn_file(
        file_path = args.pgn_file,
        num_positions = args.num_positions,
        skip_first_n_moves = args.skip_first_n_moves
    )

    # Configure logging if verbose mode is enabled.
    if args.verbose:
        get_logging_config()

    # Create output directory if it doesn't exist.
    os.makedirs(args.output_dir, exist_ok=True)

    # Set output file names.
    white_name, black_name = args.file_names

    # Log statistics if verbose mode is enabled.
    if args.verbose:
        logging.info(f"White wins: {white_games:,}")
        logging.info(f"Black wins: {black_games:,}")
        logging.info(f"Total wins: {white_games + black_games:,}")
        logging.info(f"White positions: {len(white_pos):,}")
        logging.info(f"Black positions: {len(black_pos):,}")
        logging.info(f"Total positions: {len(white_pos) + len(black_pos):,}")

    # Write positions to output files.
    write_txt(white_pos, os.path.join(args.output_dir, white_name))
    write_txt(black_pos, os.path.join(args.output_dir, black_name))


if __name__ == "__main__":
    main()