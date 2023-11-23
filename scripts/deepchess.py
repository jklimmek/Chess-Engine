import torch
import torch.nn as nn


class DeepChess(nn.Module):
    def __init__(self, pos2vec):
        super().__init__()
        self.pos2vec = pos2vec
        self.model = nn.Sequential(
            nn.Linear(200, 400),
            nn.LeakyReLU(0.3),

            nn.Linear(400, 200),
            nn.LeakyReLU(0.3),

            nn.Linear(200, 100),
            nn.LeakyReLU(0.3),

            nn.Linear(100, 2)
        )

    def forward(self, pos1, pos2):
        pos1 = self.pos2vec.encode(pos1)
        pos2 = self.pos2vec.encode(pos2)
        print(pos1.shape, pos2.shape)
        x = torch.cat([pos1, pos2], dim=1)
        x = self.model(x)
        return x
