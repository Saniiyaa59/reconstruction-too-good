"""Variational autoencoders for flat inputs: MLP and 1D conv (ECG beats), 2D conv (images).

The decoder is Gaussian with one learned scalar standard deviation sigma_x,
shared by all D input dims. Training loss is the negative ELBO:

    -ELBO(x) = SSE(x, x_hat) / (2 sigma_x^2) + D log sigma_x + D/2 log(2 pi) + beta * KL

Learning sigma_x sets the balance between reconstruction and KL from the data,
instead of fixing it by hand. Pass `fixed_sigma_x` to freeze it instead (the
robustness check in PLAN.md: sigma_x^2 = validation MSE of a trained model).
"""
import math

import torch
import torch.nn as nn

SCORES = ["recon", "kl", "neg_elbo"]  # the keys returned by GaussianVAE.score


def mlp(sizes):
    """Linear layers with ReLU between them (none after the last)."""
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class GaussianVAE(nn.Module):
    """Loss and scores shared by every VAE; subclasses define encode and decode."""

    def __init__(self, input_dim, latent_dim, beta=1.0, fixed_sigma_x=None):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
        if fixed_sigma_x is None:
            self.log_sigma_x = nn.Parameter(torch.zeros(()))  # starts at sigma_x = 1
        else:
            self.register_buffer("log_sigma_x", torch.tensor(math.log(fixed_sigma_x)))

    @staticmethod
    def reparameterize(mu, logvar):
        """z = mu + sigma * eps, so gradients flow through mu and sigma."""
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @staticmethod
    def kl_per_sample(mu, logvar):
        """Closed-form KL(N(mu, sigma^2) || N(0, I)), summed over latent dims."""
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)

    def nll_per_sample(self, x, x_hat):
        """-log p(x | z) under N(x_hat, sigma_x^2 I), summed over input dims."""
        sse = (x - x_hat).pow(2).sum(dim=1)
        D = self.input_dim
        return (sse / (2 * torch.exp(2 * self.log_sigma_x))
                + D * self.log_sigma_x + 0.5 * D * math.log(2 * math.pi))

    def loss(self, x):
        """Batch-mean negative ELBO (with this model's beta), plus parts for logging."""
        x_hat, mu, logvar = self(x)
        nll = self.nll_per_sample(x, x_hat).mean()
        kl = self.kl_per_sample(mu, logvar).mean()
        return nll + self.beta * kl, {
            "nll": nll.item(),
            "kl": kl.item(),
            "mse": (x - x_hat).pow(2).mean().item(),
            "sigma_x": self.log_sigma_x.exp().item(),
        }

    @torch.no_grad()
    def score(self, x, n_samples=10):
        """Per-sample anomaly scores.

        recon    : MSE between x and decode(mu), same units as the PCA scores
        kl       : KL of q(z|x) from the prior, how costly this input's latent code is
        neg_elbo : NLL averaged over `n_samples` draws of z, plus KL. Always uses
                   beta = 1 so it is a true bound on -log p(x), whatever beta was
                   used in training
        """
        mu, logvar = self.encode(x)
        kl = self.kl_per_sample(mu, logvar)
        nll = torch.stack([self.nll_per_sample(x, self.decode(self.reparameterize(mu, logvar)))
                           for _ in range(n_samples)]).mean(dim=0)
        return {
            "recon": (x - self.decode(mu)).pow(2).mean(dim=1),
            "kl": kl,
            "neg_elbo": nll + kl,
        }

    @torch.no_grad()
    def active_units(self, x, threshold=0.01):
        """Number of latent dims whose mean varies across inputs (Burda et al. 2016)."""
        mu, _ = self.encode(x)
        return int((mu.var(dim=0) > threshold).sum())


class MLPVAE(GaussianVAE):
    def __init__(self, input_dim, latent_dim, hidden=(128, 64), beta=1.0, fixed_sigma_x=None):
        super().__init__(input_dim, latent_dim, beta, fixed_sigma_x)
        self.encoder = mlp([input_dim, *hidden])
        self.encoder_act = nn.ReLU()
        self.to_mu = nn.Linear(hidden[-1], latent_dim)
        self.to_logvar = nn.Linear(hidden[-1], latent_dim)
        # No output activation: inputs are z-normalized, so values are unbounded
        self.decoder = mlp([latent_dim, *reversed(hidden), input_dim])

    def encode(self, x):
        h = self.encoder_act(self.encoder(x))
        return self.to_mu(h), self.to_logvar(h)

    def decode(self, z):
        return self.decoder(z)


class ConvVAE(GaussianVAE):
    """1D conv VAE: three stride-2 convolutions down, mirrored transposed convolutions up.

    There are no skip connections, so every input has to pass through the latent code.
    """

    def __init__(self, input_dim, latent_dim, channels=32, beta=1.0, fixed_sigma_x=None):
        super().__init__(input_dim, latent_dim, beta, fixed_sigma_x)
        widths = [1, channels, 2 * channels, 4 * channels]
        lengths = [input_dim]
        for _ in range(3):
            lengths.append((lengths[-1] + 1) // 2)  # stride 2, kernel 7, padding 3

        down, up = [], []
        for i in range(3):
            down += [nn.Conv1d(widths[i], widths[i + 1], 7, stride=2, padding=3), nn.ReLU()]
            # Output padding restores the exact length the matching downsampling step started from
            pad = lengths[i] - (2 * lengths[i + 1] - 1)
            up = [nn.ConvTranspose1d(widths[i + 1], widths[i], 7, stride=2, padding=3,
                                     output_padding=pad), nn.ReLU()] + up
        self.encoder = nn.Sequential(*down)
        self.decoder = nn.Sequential(*up[:-1])  # no activation on the output

        self.bottom_shape = (widths[-1], lengths[-1])
        flat = widths[-1] * lengths[-1]
        self.to_mu = nn.Linear(flat, latent_dim)
        self.to_logvar = nn.Linear(flat, latent_dim)
        self.from_latent = nn.Linear(latent_dim, flat)

    def encode(self, x):
        h = self.encoder(x.unsqueeze(1)).flatten(1)
        return self.to_mu(h), self.to_logvar(h)

    def decode(self, z):
        h = torch.relu(self.from_latent(z)).view(-1, *self.bottom_shape)
        return self.decoder(h).squeeze(1)


class ConvVAE2D(GaussianVAE):
    """2D conv VAE for flattened RGB images: three stride-2 convolutions down to a
    (size/8) x (size/8) grid, mirrored transposed convolutions up, no skip connections.

    The latent code is that grid with latent_dim / grid_cells channels, so latent_dim
    must be a multiple of the number of grid cells (64 for 64x64 images). Images come
    in flat, in (height, width, RGB) order as mvtec.py stores them.
    """

    def __init__(self, image_size, latent_dim, channels=32, beta=1.0, fixed_sigma_x=None):
        super().__init__(3 * image_size * image_size, latent_dim, beta, fixed_sigma_x)
        self.image_size = image_size
        self.grid = image_size // 8
        if latent_dim % self.grid ** 2:
            raise ValueError(f"latent_dim must be a multiple of {self.grid ** 2}")
        self.latent_channels = latent_dim // self.grid ** 2

        widths = [3, channels, 2 * channels, 4 * channels]
        down, up = [], []
        for i in range(3):
            # Kernel 4, stride 2, padding 1 halves (or doubles) the size exactly
            down += [nn.Conv2d(widths[i], widths[i + 1], 4, stride=2, padding=1), nn.ReLU()]
            up = [nn.ConvTranspose2d(widths[i + 1], widths[i], 4, stride=2, padding=1), nn.ReLU()] + up
        self.encoder = nn.Sequential(*down)
        self.decoder = nn.Sequential(*up[:-1])  # no activation on the output
        self.to_mu = nn.Conv2d(widths[-1], self.latent_channels, 1)
        self.to_logvar = nn.Conv2d(widths[-1], self.latent_channels, 1)
        self.from_latent = nn.Conv2d(self.latent_channels, widths[-1], 1)

    def encode(self, x):
        s = self.image_size
        h = self.encoder(x.view(-1, s, s, 3).permute(0, 3, 1, 2))
        return self.to_mu(h).flatten(1), self.to_logvar(h).flatten(1)

    def decode(self, z):
        h = torch.relu(self.from_latent(z.view(-1, self.latent_channels, self.grid, self.grid)))
        return self.decoder(h).permute(0, 2, 3, 1).flatten(1)
