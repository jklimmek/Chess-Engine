import torch
from torch.utils.data import Dataset

from .utils import *


class DeepChessDataset(Dataset):
    """A dataset for training DeepChess engine.

    This dataset takes a list of one-hot encoded chess positions.
    The dataset will return a tuple of the position and the label.
    The label represents which move is preffered.
    """

    def __init__(self, white_positions, black_positions):
        super().__init__()
        self.white_positions = white_positions
        self.black_positions = black_positions

    def __len__(self):
        return len(self.white_positions) + len(self.black_positions)
    
    def __getitem__(self, index):
        if index < len(self.white_positions):
            return self.white_positions[index], torch.tensor([1, 0], dtype=torch.int32)
        else:
            index -= len(self.white_positions)
            return self.black_positions[index], torch.tensor([0, 1], dtype=torch.int32)


class DBNDataset(Dataset):
    """A dataset for training DBN network.
    
    Depending on the convert_mode, this dataset will return a list of encoded chess positions.
    """

    def __init__(self, positions, convert_mode="none"):
        super().__init__()
        self.positions = positions
        self.convert_mode = convert_mode

    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, index):
        if self.convert_mode == "none":
            return self.positions[index]
        elif self.convert_mode == "fen_to_array":
            return fen_to_array(self.positions[index])