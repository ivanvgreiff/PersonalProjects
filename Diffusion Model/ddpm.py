from functools import partial

import einops as eo
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()

# A batch of (noisy) images
ImageBatch = TensorType["batch_size", "channels", "height", "width", torch.float32]

# Integer noise level between 0 and N - 1
NoiseLevel = TensorType["batch_size", torch.long]

# Normalized noise level between 0 and 1
NormalizedNoiseLevel = TensorType["batch_size", torch.float32]


def batch_broadcast(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Make `a` broadcast along the batch dimension of `b`.

    We assume the batch dimension to be the first one.
    """

    assert a.ndim == 1
    return a.view(-1, *((1,) * (b.ndim - 1)))


class ResNet(nn.Module):
    """A minimal convolutional residual network."""

    def __init__(self, feature_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()

        ConvLayer = partial(nn.Conv2d, kernel_size=3, padding=1)

        # Layers to map from data space to learned latent space and back
        self.embed = nn.Sequential(ConvLayer(feature_dim + 1, hidden_dim), nn.SiLU())
        self.out = ConvLayer(hidden_dim, feature_dim)

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    ConvLayer(hidden_dim + 1, hidden_dim),
                    nn.SiLU(),
                    ConvLayer(hidden_dim, hidden_dim, kernel_size=3),
                )
                for i in range(n_layers)
            ]
        )

    @typechecked
    def forward(self, z_n: ImageBatch, n: NormalizedNoiseLevel) -> ImageBatch:
        # Align n with the feature dimension of 2D image tensors
        n = n[:, None, None, None].expand(n.shape[0], -1, *z_n.shape[2:])

        z_n = self.embed(torch.cat((z_n, n), dim=-3))

        for layer in self.layers:
            z_n = z_n + layer(torch.cat((z_n, n), dim=-3))

        return self.out(z_n)


class MiniUnet(nn.Module):
    """A minimal U-net implementation [1].

    [1] Olaf Ronneberger, Philipp Fischer, Thomas Brox: "U-Net: Convolutional Networks
        for Biomedical Image Segmentation". https://arxiv.org/abs/1505.04597
    """

    def __init__(self, feature_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()

        assert n_layers <= 2, (
            "MNIST images can only be downsampled twice "
            "without taking care of padding issues"
        )

        self.n_layers = n_layers

        ConvLayer = partial(nn.Conv2d, kernel_size=3, padding=1)

        # Layers to map from data space to learned latent space and back
        self.embed = nn.Sequential(ConvLayer(feature_dim + 1, hidden_dim), nn.SiLU())
        self.out = ConvLayer(hidden_dim, feature_dim)

        # At each scale, we perform one nonlinear map with residual connection
        self.downscaling = nn.ModuleList(
            [
                nn.Sequential(
                    ConvLayer(4**i * hidden_dim + 1, 4**i * hidden_dim),
                    nn.SiLU(),
                    nn.Conv2d(4**i * hidden_dim, 4**i * hidden_dim, kernel_size=1),
                )
                for i in range(n_layers)
            ]
        )
        bottom_channels = 4**n_layers * hidden_dim
        self.bottom_map = nn.Sequential(
            ConvLayer(bottom_channels + 1, bottom_channels),
            nn.SiLU(),
            ConvLayer(bottom_channels, bottom_channels),
        )
        self.upscaling = nn.ModuleList(
            [
                nn.Sequential(
                    ConvLayer(2 * 4**i * hidden_dim + 1, 4**i * hidden_dim),
                    nn.SiLU(),
                    nn.Conv2d(4**i * hidden_dim, 4**i * hidden_dim, kernel_size=1),
                )
                for i in reversed(range(1, n_layers + 1))
            ]
        )

    @typechecked
    def forward(self, z_n: ImageBatch, n: NormalizedNoiseLevel) -> ImageBatch:
        # Align n with the feature dimension of 2D image tensors
        n = n[:, None, None, None]

        def cat_n(z_n, *tensors):
            return torch.cat((z_n, *tensors, n.expand(-1, -1, *z_n.shape[2:])), dim=-3)

        z_n = self.embed(cat_n(z_n))

        skip_connections = []
        for down_layer in self.downscaling:
            z_n = z_n + down_layer(cat_n(z_n))
            z_n = eo.rearrange(z_n, "b c (h h2) (w w2) -> b (c h2 w2) h w", h2=2, w2=2)
            skip_connections.append(z_n)

        z_n = self.bottom_map(cat_n(z_n))

        for up_layer in self.upscaling:
            z_n = z_n + up_layer(cat_n(z_n, skip_connections.pop()))
            z_n = eo.rearrange(z_n, "b (c h2 w2) h w -> b c (h h2) (w w2)", h2=2, w2=2)

        return self.out(z_n)


class DDPM(nn.Module):
    """A denoising diffusion model as described in [1].

    References:

    [1] "Denoising Diffusion Probabilistic Models", Ho et al., https://arxiv.org/abs/2006.11239
    """

    def __init__(self, N: int, type: str, hidden_dim: int, n_layers: int):
        """Initialize the diffusion model.

        Args:
            N: Number of diffusion steps
        """

        super().__init__()

        self.N = N
        self.type = type

        if type == "resnet":
            self.model = ResNet(feature_dim=1, hidden_dim=hidden_dim, n_layers=n_layers)
        elif type == "unet":
            self.model = MiniUnet(
                feature_dim=1, hidden_dim=hidden_dim, n_layers=n_layers
            )
        else:
            raise RuntimeError(f"Unknown model type {type}")
        
        # Compute a beta schedule and various derived variables as defined on the slides

        beta = torch.linspace(0.0001, 0.02, self.N)

        alpha = 1 - beta

        alpha_bar = torch.zeros(self.N)
        for i in range(self.N):
            alpha_bar[i] = torch.prod(alpha[:i + 1])

        beta_tilde = torch.zeros(self.N)
        for j in range(self.N):
            if j == 0:
                beta_tilde[j] = beta[j]
            else:
                beta_tilde[j] = beta[j]*(1 - alpha_bar[j - 1])/(1 - alpha_bar[j])
        ##########################################################

        self.register_buffer("alpha", alpha.float())
        self.register_buffer("beta", beta.float())
        self.register_buffer("alpha_bar", alpha_bar.float())
        self.register_buffer("beta_tilde", beta_tilde.float())

    @typechecked
    def simplified_loss(
        self, x0: ImageBatch, n: NoiseLevel, epsilon: ImageBatch
    ) -> torch.Tensor:
        """Compute the simplified ELBO loss.

        Args:
            x0: Raw image data to compute the loss for
            n: Noise level
            epsilon: Noise instance

        Returns:
            0-dimensional tensor of the fully-reduced loss
        """

        batch_size = x0.shape[0]
        i = 0
        zn = torch.zeros([batch_size, 1, 28, 28], device=x0.device)
        for k in self.alpha_bar[n]:
            zn[i, :, :, :] = x0[i, :, :, :]*torch.sqrt(k) + epsilon[i, :, :, :]*torch.sqrt(1 - k)
            i += 1
        pred_noise = self.model.forward(zn, n/self.N)
        s_loss = (epsilon - pred_noise)**2
        s_loss = s_loss.sum()/batch_size
        return s_loss


    def loss(self, x0: ImageBatch) -> torch.Tensor:
        batch_size = x0.shape[0]
        n = torch.randint(self.N, (batch_size,), device=x0.device)
        epsilon = torch.randn_like(x0, device=x0.device)

        return self.simplified_loss(x0, n, epsilon)

    @typechecked
    def estimate_x0(
        self, z_n: ImageBatch, n: NoiseLevel, epsilon: ImageBatch
    ) -> ImageBatch:
        """Re-construct x_0 from z_n and epsilon.

        Args:
            z_n: Noise images
            n: Noise level
            epsilon: Noise that produced z_n

        Returns:
            The reconstructed x_0
        """

        batch_size = z_n.shape[0]
        x0 = torch.zeros([batch_size, 1, 28, 28], z_n.device)
        i = 0
        for k in self.alpha_bar[n]:
            x0[i, :, :, :] = (z_n[i, :, :, :] - torch.sqrt(1 - k)*epsilon[i, :, :, :])/torch.sqrt(k)
            i += 1
        return x0


    @typechecked
    def sample_z_n_previous(
        self, x0: ImageBatch, z_n: ImageBatch, n: NoiseLevel
    ) -> ImageBatch:
        """Sample z_{n-1} given z_n and x_0.

        Args:
            x0: (Estimate of) images
            z_n: Noisy images
            n: Noise level

        Returns:
            A z_{n-1} sample
        """

        batch_size = x0.shape[0]
        zn_prev = torch.zeros([batch_size, 1, 28, 28], x0.device)
        i = 0
        for nn in n:
            mu_tilde = z_n[i, :, :, :]*(torch.sqrt(self.alpha[nn]))*(1 - self.alpha_bar[nn-1])/(1 - self.alpha_bar[nn])
            + x0[i, :, :, :]*self.beta[nn]*torch.sqrt(self.alpha_bar[nn-1])/(1 - self.alpha_bar[nn])
            beta_tilde = self.beta[nn]*(1 - self.alpha_bar[nn-1])/(1 - self.alpha_bar[nn])*torch.eye(28)
            zn_prev[i, :, :, :] = torch.normal(mu_tilde, beta_tilde)
            i += 1
        return zn_prev


    @torch.no_grad()
    def sample(self, batch_size: int, device: torch.device) -> ImageBatch:
        """Sample new images from scratch by iteratively denoising pure noise.

        Args:
            batch_size: Number of images to generate
            device: Device to generate them on

        Returns:
            Generated images
        """

        zn = torch.randn(batch_size, 1, 28, 28)
        zn_prev = zn
        for n in reversed(range(self.N)):
            predicted_noise = self.model.forward(zn, n*torch.ones(batch_size, dtype=torch.long)/self.N)
            x0_rec = self.estimate_x0(zn, n*torch.ones(batch_size, dtype=torch.long), predicted_noise)
            zn_prev = self.sample_z_n_previous(x0_rec, zn, n*torch.ones(batch_size, dtype=torch.long))
            zn = zn_prev
        x0_rec = self.estimate_x0(zn, n*torch.ones(batch_size, dtype=torch.long), predicted_noise)
        return x0_rec

