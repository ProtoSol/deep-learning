"""Policy evaluation and one improvement step on a three-state line."""
import numpy

GAMMA = 0.9
N_STATES = 3  # state 2 is terminal and stays at 0


def evaluate(policy, transitions, theta=1e-8):
    """Iterate the Bellman expectation equation until the values settle."""
    value = numpy.zeros(N_STATES)
    while True:
        delta = 0.0
        updated = value.copy()
        for state in range(N_STATES - 1):
            action = policy[state]
            backup = 0.0
            for prob, nxt, reward in transitions[(state, action)]:
                backup += prob * (reward + GAMMA * value[nxt])
            updated[state] = backup
            delta = max(delta, abs(updated[state] - value[state]))
        value = updated
        if delta < theta:
            return value


def improve(value, transitions):
    """Greedy one-step lookahead. Each state keeps the action with the higher q."""
    policy = {}
    states = {state for state, _action in transitions}
    for state in states:
        actions = sorted(action for s, action in transitions if s == state)
        best_action, best_q = None, None
        for action in actions:
            q = 0.0
            for prob, nxt, reward in transitions[(state, action)]:
                q += prob * (reward + GAMMA * value[nxt])
            if best_q is None or q > best_q:
                best_action, best_q = action, q
        policy[state] = best_action
    return policy


def check():
    # action 0 walks to the next state. action 1 jumps from 0 straight to the end.
    transitions = {
        (0, 0): [(1.0, 1, 0.0)],
        (0, 1): [(1.0, 2, 0.0)],
        (1, 0): [(1.0, 2, 1.0)],
    }
    value = evaluate({0: 0, 1: 0}, transitions)
    assert numpy.isclose(value[1], 1.0)
    assert numpy.isclose(value[0], GAMMA)
    assert value[2] == 0.0
    assert improve(value, transitions)[0] == 0

    transitions[(0, 1)] = [(1.0, 2, 2.0)]
    assert improve(value, transitions)[0] == 1
    jumped = evaluate({0: 1, 1: 0}, transitions)
    assert numpy.isclose(jumped[0], 2.0)
    print("policy_eval ok", round(value[0], 2), round(jumped[0], 2))


if __name__ == "__main__":
    check()
