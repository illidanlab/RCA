import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch as torch

class SingleAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super(SingleAE, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, z_dim),
        )
        self.decoder1 = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(z_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x1):
        z1 = self.encoder1(x1)
        xhat1 = self.decoder1(z1)
        return xhat1

class AE(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super(AE, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, z_dim),

        )
        self.decoder1 = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(z_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

        self.encoder2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, z_dim),

        )
        self.decoder2 = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(z_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        _, d = x1.shape
        z1 = self.encoder1(x1)
        xhat1 = self.decoder1(z1)
        z2 = self.encoder1(x2)
        xhat2 = self.decoder1(z2)
        return z1, z2, xhat1, xhat2






