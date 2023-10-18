from functools import partial

import einops as eo
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from ddpm import *

a = torch.tensor([[2.0, 2.0], [2.0, 2.0]])
b = torch.tensor([[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]])
print(torch.normal(a, b))

nd = torch.distributions.multivariate_normal.MultivariateNormal(a, b)
print(nd.sample())
"""

n = torch.randint(0, 1000, (128,))
diff = DDPM(N=1000, type="unet", hidden_dim=16, n_layers=2)
x0 = torch.rand(128, 1, 28, 28)
epsilon = torch.rand(128, 1, 28, 28)
batch_size = x0.shape[0]
z_n = torch.zeros([batch_size, 1, 28, 28])
i = 0



for n in reversed(range(diff.N)):
    predicted_noise = diff.model.forward(zn, n*torch.ones(batch_size, dtype=int)/diff.N)
    x0_rec = diff.estimate_x0(zn, n*torch.ones(batch_size, dtype=int), predicted_noise)
    zn_prev = diff.sample_z_n_previous(x0_rec, zn, n*torch.ones(batch_size, dtype=int))
    print(n)



batch_size = x0.shape[0]
zn_prev = torch.zeros([batch_size, 1, 28, 28])
i = 0
for nn in n:
    mu_tilde = z_n[i, :, :, :]*(torch.sqrt(diff.alpha[nn]))*(1 - diff.alpha_bar[nn-1])/(1 - diff.alpha_bar[nn])
    + x0[i, :, :, :]*diff.beta[nn]*torch.sqrt(diff.alpha_bar[nn-1])/(1 - diff.alpha_bar[nn])
    beta_tilde = diff.beta[nn]*(1 - diff.alpha_bar[nn-1])/(1 - diff.alpha_bar[nn])*torch.eye(28)
    print(mu_tilde)
    print(beta_tilde)
    zn_prev[i, :, :, :] = torch.normal(mu_tilde, beta_tilde)
    #print(zn_prev[i, :, :, :])
    i += 1
    if i == 3:
        break
"""