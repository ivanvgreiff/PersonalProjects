import torch
import torch.nn as nn
import torch.nn.functional as F

from typeguard import typechecked
from torchtyping import TensorType, patch_typeguard
from typing import Tuple

from vae import VAE

input_dim = 50
batch_size =50
latent_dim = 50
hidden_dim = 50

vae = VAE(input_dim, latent_dim, hidden_dim)
#encoder = VAE.Encoder(input_dim, latent_dim, hidden_dim=hidden_dim)
#decoder = VAE.Decoder(input_dim, latent_dim, hidden_dim=hidden_dim)


x = torch.randn([batch_size, latent_dim])
(mu, sigma) = vae.encoder.forward(x)
z = vae.sample_with_reparametrization(mu, sigma)
theta = vae.decoder.forward(z)
p_xz = torch.zeros([batch_size])
L = torch.zeros([batch_size])
KL = VAE.kl_divergence(mu, sigma)


print(1)



