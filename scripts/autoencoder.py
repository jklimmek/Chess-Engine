import torch.nn as nn


class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(773, 600),
            nn.BatchNorm1d(600),
            nn.LeakyReLU(0.4),
            nn.Linear(600, 400),
            nn.BatchNorm1d(400),
            nn.LeakyReLU(0.4),
            nn.Linear(400, 200),
            nn.BatchNorm1d(200),
            nn.LeakyReLU(0.4),
            nn.Linear(200, 100),
            nn.BatchNorm1d(100),
            nn.LeakyReLU(0.4),
        )
        self.decoder = nn.Sequential(
            nn.Linear(100, 200),
            nn.BatchNorm1d(200),
            nn.LeakyReLU(0.4),
            nn.Linear(200, 400),
            nn.BatchNorm1d(400),
            nn.LeakyReLU(0.4),
            nn.Linear(400, 600),
            nn.BatchNorm1d(600),
            nn.LeakyReLU(0.4),
            nn.Linear(600, 773),
            nn.BatchNorm1d(773),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
        
    def encode(self, x):
        x = self.encoder(x)
        return x
    
    def decode(self, x):
        x = self.decoder(x)
        return x