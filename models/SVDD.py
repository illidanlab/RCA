import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch as torch


class SVDD(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super(SVDD, self).__init__()
        self.c1 = torch.zeros(z_dim)
        self.R1 = None
        self.encoder = nn.Sequential(
            # nn.Dropout(0.8),
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, z_dim, bias=False),
            nn.LeakyReLU(0.1),

        )
        self.decoder = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(z_dim, hidden_dim, bias=False),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LeakyReLU(0.1),
            # nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim, bias=False),
            nn.Sigmoid(),
        )

        self.svdd_layer = nn.Linear(z_dim, 1, bias=False)

    def forward(self, x1):
        z1 = self.encoder(x1)
        xhat1 = self.decoder(z1)
        svm_output = self.svdd_layer(z1)
        return z1, xhat1, svm_output

    def init_c(self, c1):
        self.c1 = c1
        return c1

    def distance(self, z1):
        distance1 = torch.sqrt(((z1 - self.c1) ** 2).sum(dim=1))
        return distance1


class SVMLoss(torch.nn.Module):

    def __init__(self):
        super(SVMLoss, self).__init__()

    def forward(self, z1, c1):
        loss = torch.sqrt(((z1 - c1) ** 2).mean())
        return loss
