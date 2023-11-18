import torch.nn as nn


class Pos2VecBlock(nn.Module):
    def __init__(self, in_features, out_features, activation="leaky_relu", dropout=0.0):
        super().__init__()
        activation = {"leaky_relu": nn.LeakyReLU(), "sigmoid": nn.Sigmoid()}[activation]
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            activation,
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.block(x)


class Pos2Vec(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.ModuleList(
            [
                Pos2VecBlock(773, 600, "leaky_relu"),
                Pos2VecBlock(600, 400, "leaky_relu"),
                Pos2VecBlock(400, 200, "leaky_relu"),
                Pos2VecBlock(200, 100, "leaky_relu")
            ]
        )
        self.up = nn.ModuleList(
            [
                Pos2VecBlock(100, 200, "leaky_relu"),
                Pos2VecBlock(200, 400, "leaky_relu"),
                Pos2VecBlock(400, 600, "leaky_relu"),
                Pos2VecBlock(600, 773, "sigmoid")
            ]
        )

    def encode(self, x):
        for layer in self.down:
            x = layer(x)
        return x
    
    def decode(self, x):
        for layer in self.up:
            x = layer(x)
        return x
    
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
