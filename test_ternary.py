# Checks for ternary.py. Run with:  python test_ternary.py
# No model download, so it runs in a second.

from ternary import ternary_line, ternary_scale, ternary, share_of_zeros
from binary import binary_scaled, sign
from look_inside import squeeze
from race import mean_absolute_error


def close(a, b, tol=0.0001):
    return abs(a - b) < tol


def all_close(xs, ys, tol=0.0001):
    return len(xs) == len(ys) and all(close(x, y, tol) for x, y in zip(xs, ys))


CHOIR = [0.9, -0.8, 0.02, -0.01, 0.03, 0.01]


# --- The line ---

def test_line_toy():
    return close(ternary_line(CHOIR), 0.2065)


def test_line_ignores_signs():
    return close(ternary_line([1, -1, 1, -1]), 0.7)


def test_line_all_zeros():
    return ternary_line([0, 0, 0]) == 0


def test_line_empty():
    return ternary_line([]) == 0


# --- The scale s ---

def test_scale_toy():
    # Only 0.9 and 0.8 are above the line.
    return close(ternary_scale(CHOIR, 0.2065), 0.85)


def test_scale_nobody_loud():
    return ternary_scale([0.1, -0.1], 5) == 0


# --- Ternary ---

def test_ternary_toy():
    return all_close(ternary(CHOIR), [0.85, -0.85, 0, 0, 0, 0])


def test_ternary_only_three_values():
    out = ternary([0.5, -3, 0, 0.001, -0.2, 9, 0.02, -0.04])
    s = max(abs(v) for v in out)
    return set(out) <= {-s, 0, s} and len(set(out)) == 3


def test_ternary_all_zeros():
    return ternary([0, 0, 0]) == [0, 0, 0]


def test_ternary_empty():
    return ternary([]) == []


def test_ternary_all_equal_size_is_exact():
    # Every weight is loud and the same size: nothing is lost.
    w = [0.5, -0.5, 0.5, -0.5]
    return all_close(ternary(w), w)


def test_ternary_all_negative():
    # avg 2, line 1.4, loud: -2 and -3, s = 2.5
    return all_close(ternary([-1, -2, -3]), [0, -2.5, -2.5])


def test_ternary_keeps_signs_of_loud_weights():
    w = [0.01, -0.02, 0.3, -0.7, 0.05, -0.4]
    out = ternary(w)
    return all(sign(a) == sign(b) for a, b in zip(w, out) if b != 0)


def test_true_zero_stays_exact():
    # Binary must send 0 to +-s. Ternary keeps it at exactly 0.
    w = [0.6, -0.6, 0, 0.5]
    return ternary(w)[2] == 0 and binary_scaled(w)[2] != 0


def test_just_above_and_just_below_the_line():
    # [1, -1, 0.5, 0]:  avg 0.625, line 0.4375 -> 0.5 is above, stays loud.
    # [1, -1, 0.4, 0]:  avg 0.6,   line 0.42   -> 0.4 is below, goes to 0.
    above = ternary([1, -1, 0.5, 0])[2] != 0
    below = ternary([1, -1, 0.4, 0])[2] == 0
    return above and below


# --- Share of zeros ---

def test_share_of_zeros_toy():
    return close(share_of_zeros([0.85, -0.85, 0, 0, 0, 0]), 4 / 6)


def test_share_of_zeros_none():
    return share_of_zeros([1, -1]) == 0


def test_share_of_zeros_empty():
    return share_of_zeros([]) == 0


# --- The problem today's idea fixes ---

def test_ternary_quiets_the_choir():
    # Binary turns whispers into shouts. Ternary lets them stay silent.
    b_err = mean_absolute_error(CHOIR, binary_scaled(CHOIR))
    t_err = mean_absolute_error(CHOIR, ternary(CHOIR))
    return close(b_err, 0.37) and close(t_err, 0.0283) and t_err < b_err / 10


def test_ternary_beats_two_bit_on_a_crowd():
    # Same crowd as Day 23's test: a huge crowd near 0 and one rare outlier.
    w = [0.01, -0.03, 0.02, -0.02, 0.03, -0.01] * 20 + [0.7]
    return mean_absolute_error(w, ternary(w)) < mean_absolute_error(w, squeeze(w, 2))


def test_ternary_beats_binary_when_the_crowd_peaks_at_zero():
    # Like real q_lin: lots of whispers right at 0, fewer louder ones.
    # The 0 button saves the whispers, so the error roughly halves.
    whispers = [0.001, -0.002, 0.002, -0.001] * 10
    louder = [0.02, -0.03, 0.03, -0.02] * 3
    w = (whispers + louder) * 5 + [0.7]
    t_err = mean_absolute_error(w, ternary(w))
    return t_err < mean_absolute_error(w, binary_scaled(w)) / 1.5


def test_flat_crowd_gives_ternary_no_edge():
    # Honest check: sizes spread evenly (0.01, 0.02, 0.03), no real whispers.
    # The 0 button has nobody to save, and ternary even loses to binary by a hair.
    w = [0.01, -0.03, 0.02, -0.02, 0.03, -0.01] * 20 + [0.7]
    t_err = mean_absolute_error(w, ternary(w))
    b_err = mean_absolute_error(w, binary_scaled(w))
    return b_err < t_err < b_err * 1.05


def test_ternary_equals_binary_when_nobody_is_quiet():
    # Honest check: if every weight has the same size, the 0 button is unused.
    w = [0.3, -0.3, 0.3, 0.3, -0.3]
    return all_close(ternary(w), binary_scaled(w)) and share_of_zeros(ternary(w)) == 0


def test_one_big_outlier_silences_a_loud_weight():
    # Honest check: in a tiny grid, one far weight (20) drags the average up,
    # so the line jumps to about 2.4 and the loud -0.8 is sent to 0.
    w = [20, -0.8, 0.02, -0.01, 0.03, 0.01]
    return ternary(w)[1] == 0


if __name__ == "__main__":
    tests = [
        test_line_toy,
        test_line_ignores_signs,
        test_line_all_zeros,
        test_line_empty,
        test_scale_toy,
        test_scale_nobody_loud,
        test_ternary_toy,
        test_ternary_only_three_values,
        test_ternary_all_zeros,
        test_ternary_empty,
        test_ternary_all_equal_size_is_exact,
        test_ternary_all_negative,
        test_ternary_keeps_signs_of_loud_weights,
        test_true_zero_stays_exact,
        test_just_above_and_just_below_the_line,
        test_share_of_zeros_toy,
        test_share_of_zeros_none,
        test_share_of_zeros_empty,
        test_ternary_quiets_the_choir,
        test_ternary_beats_two_bit_on_a_crowd,
        test_ternary_beats_binary_when_the_crowd_peaks_at_zero,
        test_flat_crowd_gives_ternary_no_edge,
        test_ternary_equals_binary_when_nobody_is_quiet,
        test_one_big_outlier_silences_a_loud_weight,
    ]
    passed = 0
    for t in tests:
        ok = t()
        print(("PASS  " if ok else "FAIL  ") + t.__name__)
        passed += ok
    print(f"\n{passed} of {len(tests)} tests passed")
