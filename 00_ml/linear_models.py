"""Linear regression, ridge, lasso, logistic regression. NumPy only."""
import numpy


def linear_regression(x, y):
    """Normal equation. x includes a column of ones."""
    return numpy.linalg.solve(x.T @ x, x.T @ y)


def ridge(x, y, lam):
    gram = x.T @ x + lam * numpy.eye(x.shape[1])
    return numpy.linalg.solve(gram, x.T @ y)


def soft_threshold(value, lam):
    if value > lam:
        return value - lam
    if value < -lam:
        return value + lam
    return 0.0


def lasso(x, y, lam, steps=40):
    """Coordinate descent on a tiny problem. Columns of x should be centered."""
    n, p = x.shape
    w = numpy.zeros(p)
    col2 = (x ** 2).sum(axis=0)
    for _ in range(steps):
        for j in range(p):
            residual = y - x @ w + w[j] * x[:, j]
            raw = (x[:, j] @ residual) / col2[j]
            w[j] = soft_threshold(raw, lam / col2[j])
    return w


def sigmoid(z):
    return 1.0 / (1.0 + numpy.exp(-z))


def logistic(x, y, lr=0.2, steps=80):
    w = numpy.zeros(x.shape[1])
    losses = []
    for _ in range(steps):
        p = sigmoid(x @ w)
        losses.append(-numpy.mean(y * numpy.log(p + 1e-12) + (1 - y) * numpy.log(1 - p + 1e-12)))
        w = w - lr * (x.T @ (p - y)) / len(y)
    return w, losses


def check():
    x = numpy.array([[1.0, 0], [1, 1], [1, 2], [1, 3]])
    y = numpy.array([1.0, 3, 5, 7])
    w = linear_regression(x, y)
    assert numpy.allclose(w, [1, 2])

    fat = ridge(x, y, lam=10)
    assert numpy.linalg.norm(fat) < numpy.linalg.norm(w)

    # one real feature, one noise feature
    rng = numpy.random.default_rng(0)
    n = 30
    signal = rng.normal(size=n)
    noise = rng.normal(size=n)
    xl = numpy.column_stack([signal, noise])
    yl = 3 * signal
    wl = lasso(xl, yl, lam=2.0)
    assert abs(wl[0]) > abs(wl[1])

    xl = numpy.array([[1.0, -1], [1, -1], [1, 1], [1, 1]])
    yl = numpy.array([0.0, 0, 1, 1])
    wlog, losses = logistic(xl, yl)
    assert losses[-1] < losses[0]
    pred = sigmoid(xl @ wlog)
    assert pred[0] < 0.5 and pred[-1] > 0.5
    print("linear_models ok")


if __name__ == "__main__":
    check()
