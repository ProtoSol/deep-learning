"""Figures for the linear algebra chapter. Asserts mirror simple_matrix.py."""

from pathlib import Path

import matplotlib
import numpy
from matplotlib import pyplot

matplotlib.use("Agg")

OUT = Path(__file__).resolve().parents[2] / "docs" / "res" / "02"

# Same matrix as simple_matrix.py
A_EXAMPLE = numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def check_claims():
    assert A_EXAMPLE.shape == (2, 3)
    assert A_EXAMPLE[0, 0] == 1 and A_EXAMPLE[1, 2] == 6

    # A (2x3) times a 3x1 is a linear combination of the columns
    x = numpy.array([1.0, 0.0, -1.0])
    combo = A_EXAMPLE @ x
    assert numpy.allclose(combo, A_EXAMPLE[:, 0] - A_EXAMPLE[:, 2])

    # Projection of b onto a
    a = numpy.array([3.0, 0.0])
    b = numpy.array([1.0, 2.0])
    proj = (numpy.dot(a, b) / numpy.dot(a, a)) * a
    assert numpy.allclose(proj, [1.0, 0.0])

    # Least squares: y = 2x, recovered from a tall system
    xs = numpy.array([0.0, 1.0, 2.0, 3.0])
    design = numpy.column_stack([xs, numpy.ones_like(xs)])
    y = 2 * xs + 1
    slope, intercept = numpy.linalg.lstsq(design, y, rcond=None)[0]
    assert abs(slope - 2) < 1e-8 and abs(intercept - 1) < 1e-8

    # Eigenvalues of a diagonal matrix are the diagonal
    lam = numpy.linalg.eigvalsh(numpy.diag([2.0, 5.0]))
    assert numpy.allclose(sorted(lam), [2.0, 5.0])

    # Rank-1 matrix is unchanged by a rank-1 SVD truncation
    rank1 = numpy.outer(numpy.array([1.0, 2.0, 3.0]), numpy.array([1.0, -1.0]))
    u, s, vt = numpy.linalg.svd(rank1, full_matrices=False)
    recon = s[0] * numpy.outer(u[:, 0], vt[0])
    assert numpy.allclose(recon, rank1)


def _save(fig, name):
    fig.savefig(OUT / name, dpi=120)
    pyplot.close(fig)


def plot_combination():
    u = numpy.array([3.0, 1.0])
    v = numpy.array([1.0, 2.0])
    fig, ax = pyplot.subplots(figsize=(5.2, 5))
    ax.quiver(0, 0, u[0], u[1], angles="xy", scale_units="xy", scale=1, color="C0", label="u")
    ax.quiver(0, 0, v[0], v[1], angles="xy", scale_units="xy", scale=1, color="C1", label="v")
    ax.quiver(0, 0, *(u + v), angles="xy", scale_units="xy", scale=1, color="C3", label="u+v")
    ax.plot([u[0], (u + v)[0]], [u[1], (u + v)[1]], "--", color="C1", linewidth=0.8)
    ax.plot([v[0], (u + v)[0]], [v[1], (u + v)[1]], "--", color="C0", linewidth=0.8)
    ax.set_xlim(-0.5, 5)
    ax.set_ylim(-0.5, 4)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.4)
    ax.legend()
    ax.set_title("A linear combination u + v")
    fig.tight_layout()
    _save(fig, "vector_combination.png")


def plot_projection():
    a = numpy.array([3.0, 0.5])
    b = numpy.array([1.0, 2.2])
    proj = (numpy.dot(a, b) / numpy.dot(a, a)) * a
    fig, ax = pyplot.subplots(figsize=(5.2, 5))
    line = numpy.linspace(-0.2, 1.3, 20)
    ax.plot(line * a[0], line * a[1], color="gray", linewidth=0.8)
    ax.quiver(0, 0, a[0], a[1], angles="xy", scale_units="xy", scale=1, color="C0", label="a")
    ax.quiver(0, 0, b[0], b[1], angles="xy", scale_units="xy", scale=1, color="C1", label="b")
    ax.quiver(0, 0, proj[0], proj[1], angles="xy", scale_units="xy", scale=1, color="C3", label="projection")
    ax.plot([b[0], proj[0]], [b[1], proj[1]], "--", color="C3")
    ax.set_aspect("equal")
    ax.set_xlim(-0.4, 3.6)
    ax.set_ylim(-0.4, 2.8)
    ax.grid(True, alpha=0.4)
    ax.legend()
    ax.set_title("Projection of b onto the line of a")
    fig.tight_layout()
    _save(fig, "projection.png")


def plot_least_squares():
    rng = numpy.random.default_rng(0)
    xs = numpy.linspace(0, 3, 12)
    ys = 2 * xs + 1 + rng.normal(0, 0.35, size=xs.shape)
    design = numpy.column_stack([xs, numpy.ones_like(xs)])
    slope, intercept = numpy.linalg.lstsq(design, ys, rcond=None)[0]
    fig, ax = pyplot.subplots(figsize=(6.2, 3.8))
    ax.scatter(xs, ys, s=18, label="data")
    grid = numpy.linspace(0, 3, 50)
    ax.plot(grid, slope * grid + intercept, color="C3", label=f"fit {slope:.2f}x + {intercept:.2f}")
    ax.grid(True, alpha=0.4)
    ax.legend()
    ax.set_title("Least squares line")
    fig.tight_layout()
    _save(fig, "least_squares.png")


def plot_eigen():
    matrix = numpy.array([[2.0, 0.6], [0.6, 1.0]])
    lam, vecs = numpy.linalg.eigh(matrix)
    theta = numpy.linspace(0, 2 * numpy.pi, 200)
    circle = numpy.stack([numpy.cos(theta), numpy.sin(theta)])
    image = matrix @ circle
    fig, ax = pyplot.subplots(figsize=(5.2, 5))
    ax.plot(circle[0], circle[1], color="gray", label="unit circle")
    ax.plot(image[0], image[1], color="C0", label="A times the circle")
    for i in range(2):
        ax.quiver(0, 0, *(lam[i] * vecs[:, i]), angles="xy", scale_units="xy", scale=1, color="C3")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper left")
    ax.set_title("Eigenvectors stay on their own lines")
    fig.tight_layout()
    _save(fig, "eigen_ellipse.png")


def plot_svd():
    image = numpy.zeros((24, 24))
    image[4:20, 6:8] = 1
    image[4:20, 16:18] = 1
    image[10:12, 6:18] = 1
    u, s, vt = numpy.linalg.svd(image, full_matrices=False)

    def trunc(rank):
        return (u[:, :rank] * s[:rank]) @ vt[:rank]

    fig, axes = pyplot.subplots(1, 3, figsize=(7.2, 2.8))
    for ax, rank, title in zip(axes, [24, 3, 1], ["full", "rank 3", "rank 1"]):
        ax.imshow(trunc(rank), cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    _save(fig, "svd_ranks.png")


def main():
    check_claims()
    plot_combination()
    plot_projection()
    plot_least_squares()
    plot_eigen()
    plot_svd()
    print("wrote figures to", OUT)


if __name__ == "__main__":
    main()
