"""Langevin sampling from a known 2-D Gaussian, using only the score."""

import torch

MEAN = torch.tensor([1.5, -0.5])


def score(x):
    # log N(mean, I) has gradient mean - x
    return MEAN - x


def langevin(steps=200, epsilon=0.05, n=64):
    x = torch.randn(n, 2)
    for _ in range(steps):
        z = torch.randn_like(x)
        x = x + 0.5 * epsilon * score(x) + epsilon ** 0.5 * z
    return x


def check_claims():
    samples = langevin()
    empirical = samples.mean(dim=0)
    assert empirical.shape == (2,)
    assert torch.allclose(empirical, MEAN, atol=0.35)
    # one gradient step of the score at the mean is near zero
    assert torch.allclose(score(MEAN), torch.zeros(2), atol=1e-6)


if __name__ == "__main__":
    check_claims()
    print("langevin_scratch ok")
