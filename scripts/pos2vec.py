import torch.nn as nn


class Pos2VecLayer(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.linear_1 = nn.Linear(input, output)
        self.leaky_relu = nn.LeakyReLU(0.3)
        self.linear_2 = nn.Linear(output, input)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
    
    def encode(self, x):
        x = self.linear_1(x)
        x = self.leaky_relu(x)
        return x
    
    def decode(self, x):
        x = self.linear_2(x)
        x = self.tanh(x)
        return x
