import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torch.autograd import Variable
import itertools
from utils import *


class Cholesky(torch.autograd.Function):
    def forward(ctx, a):
        l = torch.cholesky(a, False)
        ctx.save_for_backward(l)
        return l

    def backward(ctx, grad_output):
        l, = ctx.saved_variables
        linv = l.inverse()
        inner = torch.tril(torch.mm(l.t(), grad_output)) * torch.tril(
            1.0 - Variable(l.data.new(l.size(1)).fill_(0.5).diag()))
        s = torch.mm(linv.t(), torch.mm(inner, linv))
        return s


class DaGMM(nn.Module):
    """Residual Block."""

    def __init__(self, input_dim, hidden_dim, z_dim, n_gmm=2):
        super(DaGMM, self).__init__()

        latent_dim = z_dim + 2  # hidden representation plus reconstruction loss and cos similarity
        # layers = []
        # layers += [nn.Linear(input_dim, hidden_dim)]
        # layers += [nn.Tanh()]
        # layers += [nn.Linear(hidden_dim, hidden_dim)]
        # layers += [nn.Tanh()]
        # layers += [nn.Linear(hidden_dim, hidden_dim)]
        # layers += [nn.Tanh()]
        # layers += [nn.Linear(hidden_dim, z_dim)]
        #
        # self.encoder = nn.Sequential(*layers)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, z_dim),

        )
        self.decoder = nn.Sequential(
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

        # layers = []
        # layers += [nn.Linear(z_dim, hidden_dim)]
        # layers += [nn.Tanh()]
        # layers += [nn.Linear(hidden_dim, hidden_dim)]
        # layers += [nn.Tanh()]
        # layers += [nn.Linear(hidden_dim, hidden_dim)]
        # layers += [nn.Tanh()]
        # layers += [nn.Linear(hidden_dim, input_dim)]
        #
        # self.decoder = nn.Sequential(*layers)

        layers = []
        layers += [nn.Linear(latent_dim, 10)]
        layers += [nn.Tanh()]
        layers += [nn.Dropout(p=0.5)]
        layers += [nn.Linear(10, n_gmm)]
        layers += [nn.Softmax(dim=1)]

        self.estimation = nn.Sequential(*layers)

        self.register_buffer("phi", torch.zeros(n_gmm))
        self.register_buffer("mu", torch.zeros(n_gmm, latent_dim))
        self.register_buffer("cov", torch.zeros(n_gmm, latent_dim, latent_dim))

    def relative_euclidean_distance(self, a, b):
        return (a - b).norm(2, dim=1) / a.norm(2, dim=1)

    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)

        rec_cosine = F.cosine_similarity(x, dec, dim=1)
        rec_euclidean = self.relative_euclidean_distance(x, dec)

        z = torch.cat([enc, rec_euclidean.unsqueeze(-1), rec_cosine.unsqueeze(-1)], dim=1)

        gamma = self.estimation(z)

        return enc, dec, z, gamma

    def compute_gmm_params(self, z, gamma):
        if torch.isnan(gamma.sum()):
            print("pause")
        gamma = torch.clamp(gamma, 0.0001, 0.9999)
        N = gamma.size(0)
        # K
        sum_gamma = torch.sum(gamma, dim=0)

        # K
        phi = (sum_gamma / N)

        self.phi = phi.data

        # K x D
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / (sum_gamma.unsqueeze(-1))
        self.mu = mu.data
        # z = N x D
        # mu = K x D
        # gamma N x K

        # z_mu = N x K x D
        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))

        # z_mu_outer = N x K x D x D
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)

        # K x D x D
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim=0) / sum_gamma.unsqueeze(-1).unsqueeze(-1)

        self.cov = cov.data

        return phi, mu, cov

    def compute_energy(self, z, phi=None, mu=None, cov=None, size_average=True):
        # Compute the energy based on the specified gmm params.
        # If none are specified use the cached values.

        if phi is None:
            phi = to_var(self.phi)
        if mu is None:
            mu = to_var(self.mu)
        if cov is None:
            cov = to_var(self.cov)
        k, D, _ = cov.size()

        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))

        cov_inverse = []
        det_cov = []
        cov_diag = 0
        eps = 1e-8
        for i in range(k):
            # K x D x D
            cov_k = cov[i] + to_var(torch.eye(D) * eps)
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))
            # (sign, logdet) = np.linalg.slogdet(cov_k.data.cpu().numpy() * (2 * np.pi))
            # det = sign * np.exp(logdet)
            # det_cov.append(det)
            det = cov_k.data.cpu().numpy() * (2 * np.pi)
            det_a = np.linalg.det(det)
            if np.isnan(np.array(det_a)):
                print('pause')
            # assert np.isnan(np.array(det_a))
            det_cov.append(np.linalg.det(cov_k.data.cpu().numpy() * (2 * np.pi)))
            cov_diag = cov_diag + torch.sum(1 / cov_k.diag())

        # K x D x D
        cov_inverse = torch.cat(cov_inverse, dim=0)
        # K
        det_cov = to_var(torch.from_numpy(np.float32(np.array(det_cov))))

        # N x K
        exp_term_tmp = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        # for stability (logsumexp)
        max_val = torch.max((exp_term_tmp).clamp(min=0), dim=1, keepdim=True)[0]

        exp_term = torch.exp(exp_term_tmp - max_val)

        sample_energy = -max_val.squeeze() - torch.log(
            torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt(det_cov)).unsqueeze(0), dim=1) + eps)

        if size_average:
            sample_energy = torch.mean(sample_energy)

        return sample_energy, cov_diag


    def loss_function(self, x, x_hat, z, gamma, lambda_energy, lambda_cov_diag):

        recon_error = torch.mean((x - x_hat) ** 2)

        phi, mu, cov = self.compute_gmm_params(z, gamma)

        sample_energy, cov_diag = self.compute_energy(z, phi, mu, cov)

        loss = recon_error + lambda_energy * sample_energy + lambda_cov_diag * cov_diag

        return loss, sample_energy, recon_error, cov_diag