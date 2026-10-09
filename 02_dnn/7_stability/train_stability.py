"""Activation scale through a deep ReLU stack. NumPy only.

ponytail: the bounds are for width 64 and depth 16. A different shape needs a
new check; the upgrade is the variance calculation in He et al. 2015.
"""
import numpy


def relu_stack(scale, depth, width, batch, rng):
    h = rng.normal(size=(width, batch))
    for _ in range(depth):
        weight = rng.normal(scale=scale, size=(width, width))
        h = numpy.maximum(0.0, weight @ h)
    return h


def check():
    depth, width, batch = 16, 64, 256
    he_scale = numpy.sqrt(2.0 / width)
    he_h = relu_stack(he_scale, depth, width, batch, numpy.random.default_rng(0))
    unit_h = relu_stack(1.0, depth, width, batch, numpy.random.default_rng(1))
    he_std = float(he_h.std())
    unit_std = float(unit_h.std())
    assert numpy.isfinite(he_h).all()
    assert 0.2 < he_std < 5.0
    assert (not numpy.isfinite(unit_std)) or unit_std > 50.0 * he_std
    shown = "inf" if not numpy.isfinite(unit_std) else f"{unit_std:.3f}"
    print(f"train_stability ok {he_std:.3f} {shown}")


if __name__ == "__main__":
    check()
