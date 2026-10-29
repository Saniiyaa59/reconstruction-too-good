"""MLP variational autoencoder for flat inputs (e.g. 140-point ECG beats).

Loss is the negative ELBO for a Gaussian decoder with fixed variance:

    -ELBO(x) = SSE(x, x_hat) + beta * KL(q(z|x) || N(0, I))

SSE = sum of squared errors over input dims. This is -log p(x|z) for a Gaussian
with variance 1/2, up to a constant. A different decoder variance only rescales
SSE relative to KL, which is the same as changing beta — so the beta sweep also
covers the choice of decoder variance.
"""
import torch
import torch.nn as nn


def mlp(sizes):
    """Linear layers with ReLU between them (none after the last)."""
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class MLPVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden=(128, 64), beta=1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta
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

    def loss(self, x):
        """Batch-mean negative ELBO, plus its two parts for logging."""
        x_hat, mu, logvar = self(x)
        sse = (x - x_hat).pow(2).sum(dim=1).mean()
        kl = self.kl_per_sample(mu, logvar).mean()
        return sse + self.beta * kl, {"sse": sse.item(), "kl": kl.item()}

    @torch.no_grad()
    def score(self, x):
        """Per-sample anomaly scores, computed deterministically from mu (no sampling).

        recon    : MSE between x and decode(mu) — same units as the PCA scores
        kl       : KL of q(z|x) from the prior — how costly this input's latent code is
        neg_elbo : SSE + beta * KL — the model's (bound on) negative log-likelihood
        """
        mu, logvar = self.encode(x)
        x_hat = self.decode(mu)
        sq = (x - x_hat).pow(2)
        kl = self.kl_per_sample(mu, logvar)
        return {
            "recon": sq.mean(dim=1),
            "kl": kl,
            "neg_elbo": sq.sum(dim=1) + self.beta * kl,
        }

    @torch.no_grad()
    def active_units(self, x, threshold=0.01):
        """Number of latent dims whose mean varies across inputs (Burda et al. 2016)."""
        mu, _ = self.encode(x)
        return int((mu.var(dim=0) > threshold).sum())
