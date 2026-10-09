"""One LSTM cell on a short random sequence. NumPy only."""
import numpy


def sigmoid(x):
    return 1.0 / (1.0 + numpy.exp(-x))


def lstm_step(x, h, c, weight, bias):
    """Four gates. forget, input, output are sigmoids; the candidate is tanh."""
    hidden = h.shape[0]
    z = weight @ numpy.concatenate([x, h]) + bias
    forget = sigmoid(z[:hidden])
    keep = sigmoid(z[hidden:2 * hidden])
    candidate = numpy.tanh(z[2 * hidden:3 * hidden])
    output = sigmoid(z[3 * hidden:])
    c = forget * c + keep * candidate
    h = output * numpy.tanh(c)
    return h, c, forget, keep, candidate, output


def check():
    hidden, features, steps = 4, 3, 5
    rng = numpy.random.default_rng(0)
    weight = rng.normal(scale=0.1, size=(4 * hidden, features + hidden))
    bias = rng.normal(scale=0.1, size=(4 * hidden,))
    h = numpy.zeros(hidden)
    c = numpy.zeros(hidden)
    for _ in range(steps):
        x = rng.normal(size=features)
        h, c, forget, keep, candidate, output = lstm_step(x, h, c, weight, bias)
        for gate in (forget, keep, output):
            assert numpy.all((gate > 0) & (gate < 1))
        assert numpy.all((candidate > -1) & (candidate < 1))
    assert h.shape == (hidden,)
    print("lstm ok", h.shape[0])


if __name__ == "__main__":
    check()
