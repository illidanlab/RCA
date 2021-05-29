import torch.nn as nn
import torch as torch
import torch.nn.functional as F

class Loss_Knn(nn.Module):

    def __init__(self, margin):
        self.margin = margin
        super(Loss_Knn, self).__init__()

    def forward(self, z, z_nn):
        n_batch, n_nn, d = z_nn.shape
        z = z.unsqueeze(dim=1).repeat(1, n_nn, 1)
        dist = ((z-z_nn)**2).mean(dim=1)
        dist = dist.sum(dim=1)
        dist = F.relu(dist-self.margin)
        loss = dist.sum(dim=0)
        return loss


class Dist_KNN(nn.Module):

    def __init__(self, margin):
        self.margin = margin
        super(Dist_KNN, self).__init__()

    def forward(self, z, z_nn):
        n_batch, n_nn, d = z_nn.shape
        z = z.unsqueeze(dim=1).repeat(1, n_nn, 1)
        dist = ((z-z_nn)**2).mean(dim=1)
        dist = dist.sum(dim=1)
        dist = F.relu(dist-self.margin)
        return dist


class VAE_LOSS(nn.Module):
    def __init__(self):
        super(VAE_LOSS, self).__init__()

    def forward(self, recon_x, x, mu, logvar, rec_type='MSE', gamma=1):
        if rec_type == 'BCE':
            BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        elif rec_type == 'MSE':
            BCE = F.mse_loss(recon_x, x, reduction='sum')
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + gamma * KLD



class VAE_LOSS_SCORE(nn.Module):
    def __init__(self):
        super(VAE_LOSS_SCORE, self).__init__()

    def forward(self, recon_x, x, mu, logvar, rec_type='MSE'):
        if rec_type == 'BCE':
            BCE = F.binary_cross_entropy(recon_x, x, reduce=False)
            BCE = BCE.sum(dim=1)
        elif rec_type == 'MSE':
            BCE = F.mse_loss(recon_x, x, reduce=False)
            BCE = BCE.sum(dim=1)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD.sum(dim=1)
        return BCE + KLD


class VAE_Outlier_SCORE(nn.Module):
    def __init__(self):
        super(VAE_Outlier_SCORE, self).__init__()

    def forward(self, recon_x, x, mu, logvar, rec_type='MSE'):
        if rec_type == 'BCE':
            BCE = F.binary_cross_entropy(recon_x, x, reduce=False)
            BCE = BCE.sum(dim=1)
        elif rec_type == 'MSE':
            BCE = F.mse_loss(recon_x, x, reduce=False)
            BCE = BCE.sum(dim=1)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        return BCE