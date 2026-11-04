# Checks for look_inside.py. Run with:  python test_look_inside.py
# These use tiny toy lists, so they run fast and need no download.

from look_inside import describe_weights, squeeze, pct_vanished


def close(a, b):
    # Decimals are never exact on a computer, so "equal" means "very close".
    return abs(a - b) < 1e-9


# --- describe_weights ---

def test_count_smallest_largest():
    facts = describe_weights([-0.5, 0.05, 0.2, 2.0])
    return facts["count"] == 4 and facts["smallest"] == -0.5 and facts["largest"] == 2.0


def test_pct_within_0_1():
    # Only 0.05 and -0.1 are within +/-0.1 -> 2 of 4 -> 50%
    facts = describe_weights([0.05, -0.1, 0.5, -3.0])
    return close(facts["pct_within_0_1"], 50)


def test_pct_within_1():
    # 0.05, -0.1, 0.5 are within +/-1 -> 3 of 4 -> 75%
    facts = describe_weights([0.05, -0.1, 0.5, -3.0])
    return close(facts["pct_within_1"], 75)


def test_pct_positive_ignores_zero():
    # 0 is neither plus nor minus -> only 1 and 2 count -> 2 of 4 -> 50%
    facts = describe_weights([0, 1, 2, -1])
    return close(facts["pct_positive"], 50)


def test_single_weight():
    facts = describe_weights([0.3])
    return facts["count"] == 1 and close(facts["pct_positive"], 100)


# --- squeeze (our Day 6 recipe on a list) ---

def test_squeeze_toy_buttons():
    # Toy from today: -3..+6 at 2 bits -> scale 3, zero point 1
    # Every value already sits on a button, so nothing changes.
    return squeeze([-3, 0, 3, 6], 2) == [-3, 0, 3, 6]


def test_squeeze_keeps_zero_exact():
    # Real 0 must come back as exactly 0 (that's the zero point's job).
    return squeeze([-0.705, 0.0, 0.496], 4)[1] == 0


def test_squeeze_error_at_most_half_step():
    # -0.705..0.496 at 4 bits -> step about 0.08 -> error at most about 0.04
    weights = [-0.705, -0.3, -0.021, 0.0191, 0.0502, 0.2, 0.496]
    step = (0.496 + 0.705) / 15
    squeezed = squeeze(weights, 4)
    # The zero-point nudge can add a tiny bit at the edges, so allow 1 full step there.
    return all(abs(w - s) <= step for w, s in zip(weights, squeezed))


def test_squeeze_today_example():
    # From today's lesson: 0.0502 -> button 10 -> about 0.08
    #                      0.0191 -> button 9  -> 0
    squeezed = squeeze([-0.705, 0.0191, 0.0502, 0.496], 4)
    return squeezed[1] == 0 and close(round(squeezed[2], 2), 0.08)


# --- pct_vanished ---

def test_vanished_toy():
    # -10..10 at 2 bits: buttons at about -10, -3.3, 3.3, 10 -> zero point lands
    # near the middle. 0.5 and -0.5 are tiny -> they vanish. 10 and -10 survive.
    return close(pct_vanished([-10, -0.5, 0.5, 10], 2), 50)


def test_vanished_more_with_fewer_bits():
    weights = [-0.705, -0.05, -0.02, 0.01, 0.03, 0.06, 0.496]
    return pct_vanished(weights, 2) >= pct_vanished(weights, 4) >= pct_vanished(weights, 8)


def test_vanished_ignores_real_zeros():
    # A weight that is already 0 didn't "vanish". Only 1.0 and -1.0 count, and both survive.
    return close(pct_vanished([-1.0, 0.0, 1.0], 8), 0)


if __name__ == "__main__":
    tests = [
        test_count_smallest_largest,
        test_pct_within_0_1,
        test_pct_within_1,
        test_pct_positive_ignores_zero,
        test_single_weight,
        test_squeeze_toy_buttons,
        test_squeeze_keeps_zero_exact,
        test_squeeze_error_at_most_half_step,
        test_squeeze_today_example,
        test_vanished_toy,
        test_vanished_more_with_fewer_bits,
        test_vanished_ignores_real_zeros,
    ]
    passed = 0
    for t in tests:
        ok = t()
        print(("PASS  " if ok else "FAIL  ") + t.__name__)
        passed += ok
    print(f"\n{passed} of {len(tests)} tests passed")
