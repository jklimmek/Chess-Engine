import torch
import torch.nn as nn


class DeepChess(nn.Module):
    def __init__(self, pos2vec):
        super().__init__()
        self.pos2vec = pos2vec
        self.model = nn.Sequential(
            nn.Linear(200, 400),
            nn.BatchNorm1d(400),
            nn.LeakyReLU(0.3),
            nn.Linear(400, 200),
            nn.BatchNorm1d(200),
            nn.LeakyReLU(0.3),
            nn.Linear(200, 100),
            nn.BatchNorm1d(100),
            nn.LeakyReLU(0.3),
            nn.Linear(100, 2),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        x1 = self.pos2vec(x1)
        x2 = self.pos2vec(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.model(x)
        return x
    
    @torch.no_grad()
    def extract(self, x):
        x = self.pos2vec(x)
        return x
    
    @torch.no_grad()
    def compare(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.model(x)
        return x
