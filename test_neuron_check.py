# Checks for neuron_check.py. Run with:  python test_neuron_check.py
# Each test prints PASS or FAIL, so you can see exactly what works.

from neuron_check import neuron, shrink

DEMO_INPUTS = [1, 2, 3]
DEMO_WEIGHTS = [0.21, -0.48, 0.73]


def close(a, b):
    # Decimals are never exact on a computer, so "equal" means "very close".
    return abs(a - b) < 1e-9


def drift(inputs, weights, bits):
    # How far the shrunk neuron's answer moves from the real one.
    return abs(neuron(inputs, shrink(weights, bits)) - neuron(inputs, weights))


def test_neuron_toy_example():
    # 1*0.5 + 2*0.25 = 1.0
    return close(neuron([1, 2], [0.5, 0.25]), 1.0)


def test_neuron_demo_answer():
    # 1*0.21 + 2*(-0.48) + 3*0.73 = 1.44
    return close(neuron(DEMO_INPUTS, DEMO_WEIGHTS), 1.44)


def test_8_bits_stays_very_close():
    return drift(DEMO_INPUTS, DEMO_WEIGHTS, 8) < 0.01


def test_more_bits_means_less_drift():
    d8 = drift(DEMO_INPUTS, DEMO_WEIGHTS, 8)
    d4 = drift(DEMO_INPUTS, DEMO_WEIGHTS, 4)
    d2 = drift(DEMO_INPUTS, DEMO_WEIGHTS, 2)
    return d8 < d4 < d2


def test_2_bits_drifts_about_0_117():
    # Only 0.21 moves (to 0.327). The smallest and biggest weights stay exact.
    return abs(drift(DEMO_INPUTS, DEMO_WEIGHTS, 2) - 0.117) < 0.001


def test_smallest_and_biggest_weights_come_back_exact():
    shrunk = shrink(DEMO_WEIGHTS, 2)
    return close(shrunk[1], -0.48) and close(shrunk[2], 0.73)


def test_errors_cancel():
    # Grocery bill: 0.4 rounds down to 0, 0.6 rounds up to 1.
    # One error is -0.4, the other +0.4, so the total stays exact.
    weights = [0, 0.4, 0.6, 3]
    return close(drift([1, 1, 1, 1], weights, 2), 0)


def test_errors_pile_up():
    # Both 0.4s round down to 0. Two errors of -0.4 push the same way: drift 0.8.
    weights = [0, 0.4, 0.4, 3]
    return close(drift([1, 1, 1, 1], weights, 2), 0.8)


def test_all_equal_weights_no_drift():
    # Edge case: 5, 5, 5 all sit on slot 0 and come back as 5.
    return close(drift(DEMO_INPUTS, [5, 5, 5], 2), 0)


def test_all_zero_weights_answer_zero():
    return close(neuron(DEMO_INPUTS, shrink([0, 0, 0], 2)), 0)


def test_zero_inputs_answer_zero():
    # Edge case: if every input is 0, the weights don't matter, even shrunk ones.
    return close(neuron([0, 0, 0], shrink(DEMO_WEIGHTS, 2)), 0)


def test_negative_inputs_same_drift_size():
    # Flipping every input's sign flips the answer, but the drift size stays the same.
    flipped = [-x for x in DEMO_INPUTS]
    return close(drift(flipped, DEMO_WEIGHTS, 2), drift(DEMO_INPUTS, DEMO_WEIGHTS, 2))


if __name__ == "__main__":
    tests = [test_neuron_toy_example, test_neuron_demo_answer,
             test_8_bits_stays_very_close, test_more_bits_means_less_drift,
             test_2_bits_drifts_about_0_117,
             test_smallest_and_biggest_weights_come_back_exact,
             test_errors_cancel, test_errors_pile_up,
             test_all_equal_weights_no_drift, test_all_zero_weights_answer_zero,
             test_zero_inputs_answer_zero, test_negative_inputs_same_drift_size]
    passed = 0
    for t in tests:
        ok = t()
        passed += ok
        print(("PASS  " if ok else "FAIL  ") + t.__name__)
    print(f"\n{passed} of {len(tests)} tests passed")
