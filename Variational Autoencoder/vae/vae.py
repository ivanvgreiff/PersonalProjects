import torch
import torch.nn as nn
import torch.nn.functional as F

from typeguard import typechecked
from torchtyping import TensorType, patch_typeguard
from typing import Tuple

from .decoder import Decoder
from .encoder import Encoder

patch_typeguard()

class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int=100):
        """Initialize the VAE model.
        
        Args:
            obs_dim (int): Dimension of the observed data x, int
            latent_dim (int): Dimension of the latent variable z, int
            hidden_dim (int): Hidden dimension of the encoder/decoder networks, int
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(input_dim, latent_dim, hidden_dim=hidden_dim)
        self.decoder = Decoder(input_dim, latent_dim, hidden_dim=hidden_dim)
    
    @typechecked
    def sample_with_reparametrization(self, mu: TensorType['batch_size', 'latent_dim'], 
                                      logsigma: TensorType['batch_size', 'latent_dim']) -> TensorType['batch_size', 'latent_dim']:
        """Draw sample from q(z) with reparametrization.
        
        We draw a single sample z_i for each data point x_i.
        
        Args:
            mu: Means of q(z) for the batch, shape [batch_size, latent_dim]
            logsigma: Log-sigmas of q(z) for the batch, shape [batch_size, latent_dim]
        
        Returns:
            z: Latent variables samples from q(z), shape [batch_size, latent_dim]
        """
        ##########################################################
        # YOUR CODE HERE
        batch_size = len(mu[:, 0])
        z = torch.zeros([batch_size, self.latent_dim])
        
        for i in range(batch_size):
            epsilon = torch.randn([self.latent_dim, 1])
            R = torch.diag( torch.exp(logsigma[i, :]) )
            z_column = torch.matmul(R, epsilon) + mu[i, :].reshape(-1, 1)

            # column vector -> row vector
            z_row = z_column.reshape(1, -1)

            z[i, :] = z_row
        ##########################################################
        return z
    
    @typechecked
    def kl_divergence(self, mu: TensorType['batch_size', 'latent_dim'], logsigma: TensorType['batch_size', 'latent_dim']) -> TensorType['batch_size']:
        """Compute KL divergence KL(q_i(z)||p(z)) for each q_i in the batch.
        
        Args:
            mu: Means of the q_i distributions, shape [batch_size, latent_dim]
            logsigma: Logarithm of standard deviations of the q_i distributions,
                      shape [batch_size, latent_dim]

        Returns:
            kl: KL divergence for each of the q_i distributions, shape [batch_size]
        """
        ##########################################################
        # YOUR CODE HERE
        batch_size = len(mu[:, 0])
        latent_dim = len(mu[0, :])
        kl = torch.zeros([batch_size])
        klx2 = torch.zeros([batch_size])
        for i in range(batch_size):
            sigma2 = torch.exp(logsigma[i, :]) * torch.exp(logsigma[i, :])
            #kl[i] = 0.5 * (torch.trace(torch.diag(sigma2)) + torch.dot(mu[i, :], mu[i, :]) - torch.det(torch.diag(sigma2)) - self.latent_dim)
            for j in range(latent_dim):
                klx2[i] += sigma2[j] + mu[i, j]*mu[i, j] - torch.log(sigma2[j]) - 1
            kl[i] = 0.5*klx2[i]
        ##########################################################
        return kl

    @typechecked
    def elbo(self, x: TensorType['batch_size', 'input_dim']) -> TensorType['batch_size']:
        """Estimate the ELBO for the mini-batch of data.
        
        Args:
            x: Mini-batch of the observations, shape [batch_size, input_dim]
        
        Returns:
            elbo_mc: MC estimate of ELBO for each sample in the mini-batch, shape [batch_size]
        """
        ##########################################################
        # YOUR CODE HERE
        batch_size = len(x[:, 0])
        input_dim = len(x[0, :])
        mu, logsigma = self.encoder.forward(x)
        z = self.sample_with_reparametrization(mu, logsigma)
        theta = self.decoder.forward(z)
        p_xz = torch.zeros([batch_size])
        elbo_mc = torch.zeros([batch_size])
        KL = self.kl_divergence(mu, logsigma)
        #BCE = torch.zeros([batch_size])
        for i in range(batch_size):
            #BCE[i] = F.binary_cross_entropy(x[i, :], theta[i, :], reduction='mean')
            for j in range(input_dim):
                p_xz[i] += torch.log( (theta[i, j] ** x[i, j]) * ((1 - theta[i, j]) ** (1 - x[i, j])) )
            elbo_mc[i] = p_xz[i] - KL[i]

        ##########################################################
        return elbo_mc

    @typechecked
    def sample(self, num_samples: int, device: str='cpu') -> Tuple[
        TensorType['num_samples', 'latent_dim'],
        TensorType['num_samples', 'input_dim'],
        TensorType['num_samples', 'input_dim']]:
        """Generate new samples from the model.
        
        Args:
            num_samples: Number of samples to generate.
        
        Returns:
            z: Sampled latent codes, shape [num_samples, latent_dim]
            theta: Parameters of the output distribution, shape [num_samples, input_dim]
            x: Corresponding samples generated by the model, shape [num_samples, input_dim]
        """
        ##########################################################
        # YOUR CODE HERE
        z = torch.randn([num_samples, self.latent_dim])
        theta = self.decoder.forward(z)
        x = torch.bernoulli(theta)        
        ##########################################################
        return (z, theta, x)
