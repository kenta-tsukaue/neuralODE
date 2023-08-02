import torch
from torch import nn
from torchdiffeq import odeint

class ODEF(nn.Module):
    def __init__(self, dim):
        super(ODEF, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 50),
            nn.ReLU(),
            nn.Linear(50, dim),
        )

    def forward(self, t, x):
        return self.net(x)

class NeuralODE(nn.Module):
    def __init__(self, output_dim):
        super(NeuralODE, self).__init__()
        self.ode = ODEF(output_dim)

    def forward(self, x):
        out = odeint(self.ode, x, torch.tensor([0, 1.]), method='dopri5')
        return out[-1]

class ImageClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ImageClassifier, self).__init__()
        self.downsampling = nn.Sequential(
            nn.Conv2d(input_dim, 64, 3, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1),
        )
        self.neural_ode = NeuralODE(64 * 7 * 7)
        self.upsampling = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, x):
        x = self.downsampling(x)
        x = x.view(x.size(0), -1)
        x = self.neural_ode(x)
        x = self.upsampling(x)
        return x