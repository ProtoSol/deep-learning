"""Vision Transformer encoder from scratch. No nn.TransformerEncoder."""

import math

import torch

IMG = 28
PATCH = 7
N_PATCH = (IMG // PATCH) ** 2
DIM = 32
HEADS = 4
HEAD_DIM = DIM // HEADS
DEPTH = 2
N_CLASS = 10


def patchify(images):
    """Split (B, 1, 28, 28) into (B, 16, 49) patches."""
    batch = images.shape[0]
    patches = images.unfold(2, PATCH, PATCH).unfold(3, PATCH, PATCH)
    return patches.contiguous().view(batch, N_PATCH, PATCH * PATCH)


def scaled_dot_product(query, key, value):
    scale = math.sqrt(query.shape[-1])
    scores = query @ key.transpose(-2, -1) / scale
    weights = torch.softmax(scores, dim=-1)
    return weights @ value, weights


class EncoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = torch.nn.Linear(DIM, 3 * DIM)
        self.proj = torch.nn.Linear(DIM, DIM)
        self.norm1 = torch.nn.LayerNorm(DIM)
        self.norm2 = torch.nn.LayerNorm(DIM)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(DIM, 4 * DIM),
            torch.nn.GELU(),
            torch.nn.Linear(4 * DIM, DIM),
        )

    def forward(self, tokens):
        batch, seq, _ = tokens.shape
        residual = tokens
        qkv = self.qkv(self.norm1(tokens))
        query, key, value = qkv.chunk(3, dim=-1)
        query = query.view(batch, seq, HEADS, HEAD_DIM).transpose(1, 2)
        key = key.view(batch, seq, HEADS, HEAD_DIM).transpose(1, 2)
        value = value.view(batch, seq, HEADS, HEAD_DIM).transpose(1, 2)
        attended, _ = scaled_dot_product(query, key, value)
        attended = attended.transpose(1, 2).contiguous().view(batch, seq, DIM)
        tokens = residual + self.proj(attended)
        return tokens + self.ff(self.norm2(tokens))


class VisionTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_proj = torch.nn.Linear(PATCH * PATCH, DIM)
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, DIM))
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, N_PATCH + 1, DIM))
        self.layers = torch.nn.ModuleList(EncoderLayer() for _ in range(DEPTH))
        self.norm = torch.nn.LayerNorm(DIM)
        self.head = torch.nn.Linear(DIM, N_CLASS)

    def forward(self, images):
        batch = images.shape[0]
        tokens = self.patch_proj(patchify(images))
        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed
        for layer in self.layers:
            tokens = layer(tokens)
        return self.head(self.norm(tokens)[:, 0])


def check_claims():
    assert IMG % PATCH == 0
    assert N_PATCH == 16
    assert DIM % HEADS == 0

    images = torch.randn(3, 1, IMG, IMG)
    patches = patchify(images)
    assert patches.shape == (3, 16, 49)

    model = VisionTransformer()
    logits = model(images)
    assert logits.shape == (3, N_CLASS)

    query = torch.randn(2, HEADS, 5, HEAD_DIM)
    out, weights = scaled_dot_product(query, query, query)
    assert out.shape == query.shape
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, HEADS, 5), atol=1e-5)

    labels = torch.randint(0, N_CLASS, (3,))
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    assert model.head.weight.grad is not None


if __name__ == "__main__":
    check_claims()
    print("vit_scratch ok", N_PATCH, "patches")
