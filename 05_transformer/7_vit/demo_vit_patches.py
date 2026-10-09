"""Patch-grid figure for the Vision Transformer page. Asserts 28/7 -> 16 patches."""

from pathlib import Path

import matplotlib
import numpy
from matplotlib import pyplot

matplotlib.use("Agg")

OUT = Path(__file__).resolve().parents[2] / "docs" / "res" / "05"
IMAGE_SIZE = 28
PATCH_SIZE = 7


def check_claims():
    assert IMAGE_SIZE % PATCH_SIZE == 0
    n = IMAGE_SIZE // PATCH_SIZE
    assert n * n == 16


def make_digit():
    img = numpy.zeros((IMAGE_SIZE, IMAGE_SIZE))
    # a crude '7' so the figure does not need torchvision
    img[3:6, 6:22] = 1.0
    for i in range(18):
        r = 6 + i
        c = 20 - int(i * 0.7)
        img[r : r + 2, c : c + 3] = 1.0
    return img


def plot_patches():
    img = make_digit()
    n = IMAGE_SIZE // PATCH_SIZE
    fig, axes = pyplot.subplots(1, 2, figsize=(7.2, 3.4))
    axes[0].imshow(img, cmap="gray_r")
    for i in range(1, n):
        axes[0].axhline(i * PATCH_SIZE - 0.5, color="C3", linewidth=0.8)
        axes[0].axvline(i * PATCH_SIZE - 0.5, color="C3", linewidth=0.8)
    axes[0].set_title("image cut into 7x7 patches")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    k = 0
    canvas = numpy.ones((n * (PATCH_SIZE + 1) - 1, n * (PATCH_SIZE + 1) - 1))
    for r in range(n):
        for c in range(n):
            patch = img[r * PATCH_SIZE : (r + 1) * PATCH_SIZE, c * PATCH_SIZE : (c + 1) * PATCH_SIZE]
            rr = r * (PATCH_SIZE + 1)
            cc = c * (PATCH_SIZE + 1)
            canvas[rr : rr + PATCH_SIZE, cc : cc + PATCH_SIZE] = patch
            k += 1
    axes[1].imshow(canvas, cmap="gray_r")
    axes[1].set_title("sequence of 16 patches")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "vit_patches.png", dpi=120)
    pyplot.close(fig)
    assert k == 16


if __name__ == "__main__":
    check_claims()
    plot_patches()
    print("wrote", OUT / "vit_patches.png")
