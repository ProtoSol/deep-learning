"""Self, causal, and cross attention from one scaled-dot-product kernel."""

import math

import torch


def scaled_dot_product(query, key, value, mask=None):
    scale = math.sqrt(query.shape[-1])
    scores = query @ key.transpose(-2, -1) / scale
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ value, weights


def causal_mask(length):
    return torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)


def check_claims():
    batch, heads, tokens, dim = 2, 2, 4, 8
    x = torch.randn(batch, heads, tokens, dim)
    context = torch.randn(batch, heads, 6, dim)

    self_out, self_w = scaled_dot_product(x, x, x)
    assert self_out.shape == x.shape
    assert torch.allclose(self_w.sum(-1), torch.ones(batch, heads, tokens), atol=1e-5)

    mask = causal_mask(tokens)
    causal_out, causal_w = scaled_dot_product(x, x, x, mask=mask)
    assert causal_out.shape == x.shape
    # position 0 can only see itself
    assert torch.allclose(causal_w[:, :, 0, 0], torch.ones(batch, heads), atol=1e-5)
    assert torch.allclose(causal_w[:, :, 0, 1:], torch.zeros(batch, heads, tokens - 1), atol=1e-5)

    cross_out, cross_w = scaled_dot_product(x, context, context)
    assert cross_out.shape == x.shape
    assert cross_w.shape == (batch, heads, tokens, 6)


if __name__ == "__main__":
    check_claims()
    print("attention_kinds ok")
