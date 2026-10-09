"""Figures for the calculus chapter. One assert checks each numeric claim."""

from pathlib import Path

import matplotlib
import numpy
from matplotlib import pyplot
from matplotlib.animation import FuncAnimation, PillowWriter

matplotlib.use("Agg")

OUT = Path(__file__).resolve().parents[2] / "docs" / "res" / "01"


def f_square(x):
    return x ** 2


def _slope(fn, x, h=1e-5):
    return (fn(x + h) - fn(x - h)) / (2 * h)


def check_claims():
    # power rule: d/dx x^2 at x=3 is 6
    assert abs(_slope(lambda x: x ** 2, 3.0) - 6.0) < 1e-4
    # constant multiple: d/dx 4x = 4
    assert abs(_slope(lambda x: 4 * x, 2.0) - 4.0) < 1e-4
    # chain rule: d/dx sin(x^2) = cos(x^2) * 2x, at x=1
    chain = numpy.cos(1.0) * 2
    assert abs(_slope(lambda x: numpy.sin(x ** 2), 1.0) - chain) < 1e-4
    # product rule: d/dx (x sin x) = sin x + x cos x, at pi/2 equals 1
    product = numpy.sin(numpy.pi / 2) + (numpy.pi / 2) * numpy.cos(numpy.pi / 2)
    assert abs(product - 1.0) < 1e-6
    assert abs(_slope(lambda x: x * numpy.sin(x), numpy.pi / 2) - 1.0) < 1e-4
    # limit of (x^2 - 1) / (x - 1) as x -> 1 is 2
    xs = numpy.array([1.001, 0.999, 1.0001])
    vals = (xs ** 2 - 1) / (xs - 1)
    assert numpy.allclose(vals, 2.0, atol=1e-2)
    # Newton on f(x)=x^2-2 from 1 lands near sqrt(2)
    guess = 1.0
    for _ in range(6):
        guess = guess - (guess ** 2 - 2) / (2 * guess)
    assert abs(guess - 2 ** 0.5) < 1e-4


def _save(fig, name):
    fig.savefig(OUT / name, dpi=120)
    pyplot.close(fig)


def plot_trig():
    x = numpy.linspace(-2 * numpy.pi, 2 * numpy.pi, 400)
    fig, ax = pyplot.subplots(figsize=(7, 3.6))
    ax.plot(x, numpy.sin(x), label="sin x")
    ax.plot(x, numpy.cos(x), label="cos x")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks([-2 * numpy.pi, -numpy.pi, 0, numpy.pi, 2 * numpy.pi])
    ax.set_xticklabels([r"$-2\pi$", r"$-\pi$", "0", r"$\pi$", r"$2\pi$"])
    ax.set_ylim(-1.4, 1.4)
    ax.legend()
    ax.grid(True, alpha=0.4)
    ax.set_title("Sine and cosine")
    fig.tight_layout()
    _save(fig, "trig_functions.png")


def plot_inverse():
    x = numpy.linspace(0.15, 2.2, 200)
    fig, ax = pyplot.subplots(figsize=(5.5, 5))
    ax.plot(x, numpy.exp(x - 1), label=r"$f(x)=e^{x-1}$")
    ax.plot(numpy.exp(x - 1), x, label=r"$f^{-1}(x)=1+\ln x$")
    line = numpy.linspace(0.2, 3.2, 50)
    ax.plot(line, line, "--", color="gray", label="y = x")
    ax.set_xlim(0, 3.4)
    ax.set_ylim(0, 3.4)
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.4)
    ax.set_title("A function and its inverse")
    fig.tight_layout()
    _save(fig, "inverse_function.png")


def plot_exp_log():
    x = numpy.linspace(-2, 2, 200)
    fig, ax = pyplot.subplots(figsize=(6.5, 3.8))
    ax.plot(x, numpy.exp(x), label=r"$e^{x}$")
    positive = numpy.linspace(0.05, 4, 200)
    ax.plot(positive, numpy.log(positive), label=r"$\ln x$")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_ylim(-2.5, 4)
    ax.legend()
    ax.grid(True, alpha=0.4)
    ax.set_title("Exponential and natural log")
    fig.tight_layout()
    _save(fig, "exp_log.png")


def plot_limit():
    x_left = numpy.linspace(-1, 0.98, 80)
    x_right = numpy.linspace(1.02, 3, 80)
    fig, ax = pyplot.subplots(figsize=(6.5, 3.8))
    ax.plot(x_left, x_left + 1, color="C0")
    ax.plot(x_right, x_right + 1, color="C0", label=r"$f(x)=(x^2-1)/(x-1)$")
    ax.scatter([1], [2], facecolors="none", edgecolors="C0", s=60, zorder=3, label="hole at x=1")
    ax.axvline(1, color="gray", linestyle=":", linewidth=0.8)
    ax.legend()
    ax.grid(True, alpha=0.4)
    ax.set_title("The limit is 2, even though f(1) is undefined")
    fig.tight_layout()
    _save(fig, "limit_hole.png")


def plot_derivative_pair():
    x = numpy.linspace(-2, 3, 200)
    fig, ax = pyplot.subplots(figsize=(6.5, 3.8))
    ax.plot(x, f_square(x), label=r"$f(x)=x^2$")
    ax.plot(x, 2 * x, label=r"$f'(x)=2x$")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.legend()
    ax.grid(True, alpha=0.4)
    ax.set_title("A function and its derivative")
    fig.tight_layout()
    _save(fig, "derivative_function.png")


def plot_maxima():
    x = numpy.linspace(-2, 3, 300)
    y = x ** 3 - 3 * x ** 2 + 1  # y' = 3x^2 - 6x = 3x(x-2), crit at 0 and 2
    fig, ax = pyplot.subplots(figsize=(6.5, 3.8))
    ax.plot(x, y, label=r"$f(x)=x^3-3x^2+1$")
    ax.scatter([0, 2], [1, -3], color="C3", zorder=3, label="critical points")
    ax.legend()
    ax.grid(True, alpha=0.4)
    ax.set_title("Local max at x=0, local min at x=2")
    fig.tight_layout()
    _save(fig, "maxima_minima.png")


def plot_linear_approx():
    x = numpy.linspace(0.2, 3, 200)
    a = 1.0
    fig, ax = pyplot.subplots(figsize=(6.5, 3.8))
    ax.plot(x, numpy.exp(x - 1), label=r"$f(x)=e^{x-1}$")
    # f(1)=1, f'(x)=e^{x-1}, f'(1)=1, L(x)=1+(x-1)
    ax.plot(x, 1 + (x - a), label="tangent at x=1")
    ax.scatter([1], [1], color="C3", zorder=3)
    ax.legend()
    ax.grid(True, alpha=0.4)
    ax.set_title("Linear approximation")
    fig.tight_layout()
    _save(fig, "linear_approx.png")


def plot_riemann():
    a, b, n = 0.0, 2.0, 8
    xs = numpy.linspace(a, b, n + 1)
    fig, ax = pyplot.subplots(figsize=(6.5, 3.8))
    curve_x = numpy.linspace(a, b, 200)
    ax.plot(curve_x, f_square(curve_x), color="C0", label=r"$f(x)=x^2$")
    for left in xs[:-1]:
        ax.add_patch(pyplot.Rectangle((left, 0), (b - a) / n, f_square(left),
                                      facecolor="C0", alpha=0.25, edgecolor="C0"))
    ax.set_xlim(a, b)
    ax.set_ylim(0, 4.4)
    ax.legend()
    ax.grid(True, alpha=0.4)
    ax.set_title("A Riemann sum estimates the area")
    fig.tight_layout()
    _save(fig, "riemann_sum.png")


def animate_secant():
    a = 1.0
    xs = numpy.linspace(-0.4, 2.6, 200)
    hs = numpy.linspace(1.4, 0.08, 28)
    fig, ax = pyplot.subplots(figsize=(6.5, 3.8))
    ax.plot(xs, f_square(xs), color="C0", label=r"$f(x)=x^2$")
    slope_t = 2 * a
    tangent, = ax.plot(xs, f_square(a) + slope_t * (xs - a), color="C2", label="tangent")
    secant, = ax.plot([], [], color="C3", label="secant")
    point, = ax.plot([], [], "o", color="C3")
    text = ax.text(0.02, 0.92, "", transform=ax.transAxes)
    ax.set_ylim(-0.5, 7)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.4)
    ax.set_title("Secant slopes approach the derivative")

    def update(i):
        h = hs[i]
        slope = (f_square(a + h) - f_square(a)) / h
        secant.set_data(xs, f_square(a) + slope * (xs - a))
        point.set_data([a, a + h], [f_square(a), f_square(a + h)])
        text.set_text(f"h = {h:.2f}    slope = {slope:.2f}    f'(1) = 2")
        return secant, point, text

    anim = FuncAnimation(fig, update, frames=len(hs), interval=120, blit=True)
    anim.save(OUT / "secant_tangent.gif", writer=PillowWriter(fps=8))
    pyplot.close(fig)
    # tangent is drawn so the still frame matches the last idea; silence unused
    assert tangent is not None


def animate_chain():
    xs = numpy.linspace(-2.2, 2.2, 240)
    inner = xs ** 2
    outer = numpy.sin(inner)
    frames = numpy.linspace(-1.6, 1.6, 32)
    fig, axes = pyplot.subplots(1, 2, figsize=(8.2, 3.6))
    axes[0].plot(xs, inner, color="C0")
    axes[0].set_title(r"inner: $u=x^2$")
    axes[1].plot(xs, outer, color="C1")
    axes[1].set_title(r"composition: $y=\sin(x^2)$")
    for ax in axes:
        ax.grid(True, alpha=0.4)
        ax.axhline(0, color="black", linewidth=0.5)
    dot0, = axes[0].plot([], [], "o", color="C3")
    dot1, = axes[1].plot([], [], "o", color="C3")
    note = fig.text(0.5, 0.02, "", ha="center")

    def update(i):
        x = frames[i]
        u = x ** 2
        y = numpy.sin(u)
        slope = numpy.cos(u) * 2 * x
        dot0.set_data([x], [u])
        dot1.set_data([x], [y])
        note.set_text(f"x = {x:.2f}    dy/dx = cos(x^2) * 2x = {slope:.2f}")
        return dot0, dot1

    anim = FuncAnimation(fig, update, frames=len(frames), interval=120, blit=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    anim.save(OUT / "chain_rule.gif", writer=PillowWriter(fps=8))
    pyplot.close(fig)


def main():
    check_claims()
    plot_trig()
    plot_inverse()
    plot_exp_log()
    plot_limit()
    plot_derivative_pair()
    plot_maxima()
    plot_linear_approx()
    plot_riemann()
    animate_secant()
    animate_chain()
    print("wrote figures to", OUT)


if __name__ == "__main__":
    main()
