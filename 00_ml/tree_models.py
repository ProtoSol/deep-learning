"""Depth-2 tree, bagging, a tiny random forest, and AdaBoost. NumPy only."""
import numpy


def gini(y):
    if len(y) == 0:
        return 0.0
    p = numpy.mean(y)
    return 2 * p * (1 - p)


def best_split(x, y, feats):
    best = None
    for j in feats:
        cuts = numpy.unique(x[:, j])
        for cut in cuts[:-1]:
            left = y[x[:, j] <= cut]
            right = y[x[:, j] > cut]
            if len(left) == 0 or len(right) == 0:
                continue
            score = (len(left) * gini(left) + len(right) * gini(right)) / len(y)
            if best is None or score < best[0]:
                best = (score, j, cut)
    return best


def tree(x, y, depth, feats=None):
    if feats is None:
        feats = list(range(x.shape[1]))
    pred = 1 if numpy.mean(y) >= 0.5 else 0
    if depth == 0 or len(numpy.unique(y)) == 1:
        return ("leaf", pred)
    found = best_split(x, y, feats)
    if found is None:
        return ("leaf", pred)
    _score, j, cut = found
    left = x[:, j] <= cut
    return ("node", j, cut,
            tree(x[left], y[left], depth - 1, feats),
            tree(x[~left], y[~left], depth - 1, feats))


def predict_one(model, row):
    if model[0] == "leaf":
        return model[1]
    _tag, j, cut, left, right = model
    return predict_one(left if row[j] <= cut else right, row)


def predict(model, x):
    return numpy.array([predict_one(model, row) for row in x])


def bagging(x, y, n_trees, depth, rng, feature_fraction=1.0):
    models = []
    n, p = x.shape
    for _ in range(n_trees):
        idx = rng.integers(0, n, size=n)
        k = max(1, int(p * feature_fraction))
        feats = list(rng.choice(p, size=k, replace=False))
        models.append(tree(x[idx], y[idx], depth, feats))
    return models


def vote(models, x):
    votes = numpy.mean([predict(m, x) for m in models], axis=0)
    return (votes >= 0.5).astype(int)


def stump(x, y, weights):
    """One-feature threshold that minimizes the weighted error."""
    best = None
    for j in range(x.shape[1]):
        for cut in numpy.unique(x[:, j])[:-1]:
            for flip in (0, 1):
                pred = numpy.where(x[:, j] <= cut, flip, 1 - flip)
                err = numpy.sum(weights * (pred != y))
                if best is None or err < best[0]:
                    best = (err, j, cut, flip)
    return best


def adaboost(x, y, rounds=8):
    n = len(y)
    w = numpy.ones(n) / n
    alphas, stumps = [], []
    for _ in range(rounds):
        err, j, cut, flip = stump(x, y, w)
        err = numpy.clip(err, 1e-6, 1 - 1e-6)
        alpha = 0.5 * numpy.log((1 - err) / err)
        pred = numpy.where(x[:, j] <= cut, flip, 1 - flip)
        signed = numpy.where(pred == y, 1.0, -1.0)
        w = w * numpy.exp(-alpha * signed)
        w = w / w.sum()
        alphas.append(alpha)
        stumps.append((j, cut, flip))
    return alphas, stumps


def adaboost_predict(alphas, stumps, x):
    score = numpy.zeros(len(x))
    for alpha, (j, cut, flip) in zip(alphas, stumps):
        pred = numpy.where(x[:, j] <= cut, flip, 1 - flip)
        score += alpha * numpy.where(pred == 1, 1.0, -1.0)
    return (score >= 0).astype(int)


def check():
    # XOR needs both features and depth 2
    x = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = numpy.array([0, 1, 1, 0])
    model = tree(x, y, depth=2)
    assert numpy.array_equal(predict(model, x), y)
    shallow = tree(x, y, depth=1)
    assert not numpy.array_equal(predict(shallow, x), y)

    # Same toy set for bagging, a one-feature forest, and AdaBoost.
    # The label is the sign of the first column; the second column is noise.
    rng = numpy.random.default_rng(0)
    x_toy = numpy.column_stack([
        numpy.array([-1.0] * 12 + [1.0] * 12),
        rng.normal(size=24),
    ])
    y_toy = numpy.array([0] * 12 + [1] * 12)
    bag = bagging(x_toy, y_toy, n_trees=20, depth=2, rng=rng)
    assert numpy.array_equal(vote(bag, x_toy), y_toy)
    forest = bagging(x_toy, y_toy, n_trees=80, depth=2, rng=rng, feature_fraction=0.5)
    assert numpy.array_equal(vote(forest, x_toy), y_toy)

    alphas, stumps = adaboost(x_toy, y_toy, rounds=8)
    assert numpy.array_equal(adaboost_predict(alphas, stumps, x_toy), y_toy)
    print("tree_models ok")


if __name__ == "__main__":
    check()
