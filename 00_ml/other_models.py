"""kNN, Gaussian naive Bayes, linear SVM, k-means, and a train/test split."""
import numpy


def knn(train_x, train_y, query, k=3):
    d = numpy.linalg.norm(train_x - query, axis=1)
    nearest = train_y[numpy.argsort(d)[:k]]
    return 1 if numpy.mean(nearest) >= 0.5 else 0


def gaussian_nb(train_x, train_y, query):
    classes = numpy.unique(train_y)
    best_c, best_log = None, None
    for c in classes:
        rows = train_x[train_y == c]
        mean = rows.mean(axis=0)
        var = rows.var(axis=0) + 1e-6
        logp = numpy.log(len(rows) / len(train_y))
        logp -= 0.5 * numpy.sum(numpy.log(2 * numpy.pi * var) + (query - mean) ** 2 / var)
        if best_log is None or logp > best_log:
            best_c, best_log = c, logp
    return best_c


def hinge_svm(x, y, lr=0.05, steps=200, lam=0.01):
    """y in {-1, +1}. Subgradient of mean hinge plus a small L2 term."""
    w = numpy.zeros(x.shape[1])
    for _ in range(steps):
        margin = y * (x @ w)
        active = margin < 1
        grad = numpy.zeros_like(w)
        if numpy.any(active):
            grad = -(x[active].T @ y[active]) / len(y)
        w = w - lr * (grad + lam * w)
    return w


def kmeans(x, k, steps=20, rng=None):
    rng = numpy.random.default_rng(0) if rng is None else rng
    centers = x[rng.choice(len(x), size=k, replace=False)].copy()
    labels = numpy.zeros(len(x), dtype=int)
    for _ in range(steps):
        d = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = d.argmin(axis=1)
        for j in range(k):
            if numpy.any(labels == j):
                centers[j] = x[labels == j].mean(axis=0)
    return labels, centers


def accuracy(pred, y):
    return float(numpy.mean(pred == y))


def train_test_split(x, y, frac=0.5, rng=None):
    rng = numpy.random.default_rng(0) if rng is None else rng
    idx = rng.permutation(len(y))
    cut = int(len(y) * frac)
    return x[idx[:cut]], y[idx[:cut]], x[idx[cut:]], y[idx[cut:]]


def check():
    rng = numpy.random.default_rng(1)
    a = rng.normal(loc=0, scale=0.3, size=(20, 2))
    b = rng.normal(loc=3, scale=0.3, size=(20, 2))
    x = numpy.vstack([a, b])
    y = numpy.array([0] * 20 + [1] * 20)
    train_x, train_y, test_x, test_y = train_test_split(x, y, rng=rng)
    knn_pred = numpy.array([knn(train_x, train_y, row, k=3) for row in test_x])
    nb_pred = numpy.array([gaussian_nb(train_x, train_y, row) for row in test_x])
    assert accuracy(knn_pred, test_y) == 1.0
    assert accuracy(nb_pred, test_y) == 1.0

    y_pm = numpy.where(y == 1, 1.0, -1.0)
    w = hinge_svm(numpy.column_stack([numpy.ones(len(x)), x]), y_pm)
    pred = numpy.sign(numpy.column_stack([numpy.ones(len(x)), x]) @ w)
    assert accuracy(pred, y_pm) == 1.0

    labels, centers = kmeans(x, k=2, rng=rng)
    # each true class should land mostly in one cluster
    for c in (0, 1):
        votes = labels[y == c]
        assert max(numpy.mean(votes == 0), numpy.mean(votes == 1)) > 0.9
    assert centers.shape == (2, 2)
    print("other_models ok", round(accuracy(knn_pred, test_y), 2))


if __name__ == "__main__":
    check()
