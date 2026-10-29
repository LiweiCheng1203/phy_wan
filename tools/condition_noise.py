"""Gaussian perturbation of the first-frame condition latent during training."""
import torch


def apply_condition_noise(latent, generator, prob, sigma_max):
    """Perturb a normalized VAE condition latent [B, C, 1, H, W] independently per sample.

    A sample is noised with probability `prob` at sigma ~ U(0, sigma_max). The same random values
    are drawn whether or not a sample is noised, so the generator stream stays deterministic.
    """
    batch, device = latent.shape[0], latent.device
    noised = torch.rand(batch, generator=generator, device=device) < prob
    sigma = torch.rand(batch, generator=generator, device=device) * sigma_max * noised
    noise = torch.randn(latent.shape, generator=generator, device=device, dtype=torch.float32)
    output = latent.float() + sigma.view(-1, *[1] * (latent.ndim - 1)) * noise
    return output.to(latent.dtype), sigma
