# Checks for binary.py. Run with:  python test_binary.py
# No model download, so it runs in a second.

from binary import sign, binary_plain, average_size, binary_scaled, neuron_answer
from look_inside import squeeze
from race import mean_absolute_error


def close(a, b, tol=0.0001):
    return abs(a - b) < tol


def all_close(xs, ys, tol=0.0001):
    return len(xs) == len(ys) and all(close(x, y, tol) for x, y in zip(xs, ys))


# --- Sign ---

def test_sign_positive():
    return sign(0.73) == 1 and sign(5) == 1


def test_sign_negative():
    return sign(-0.48) == -1 and sign(-0.0001) == -1


def test_sign_zero_goes_to_plus_one():
    # 0 has no sign; we pick +1 so only two values exist.
    return sign(0) == 1


# --- Plain binary (signs only) ---

def test_plain_toy():
    return binary_plain([0.21, -0.48, 0.73]) == [1, -1, 1]


def test_plain_only_two_values():
    out = binary_plain([0.5, -3, 0, 0.001, -0.2, 9])
    return set(out) == {1, -1}


def test_plain_all_zeros():
    return binary_plain([0, 0, 0]) == [1, 1, 1]


# --- Average size ---

def test_average_size_toy():
    return close(average_size([0.21, -0.48, 0.73]), 0.4733)


def test_average_size_ignores_signs():
    # With signs, these would average to 0. Sizes average to 2.
    return close(average_size([2, -2, 2, -2]), 2)


def test_average_size_all_zeros():
    return average_size([0, 0, 0]) == 0


def test_average_size_empty():
    return average_size([]) == 0


# --- Scaled binary ---

def test_scaled_toy():
    return all_close(binary_scaled([0.21, -0.48, 0.73]), [0.4733, -0.4733, 0.4733])


def test_scaled_only_two_values():
    out = binary_scaled([0.1, -0.3, 0.2, -0.4])
    return set(out) == {0.25, -0.25}


def test_scaled_all_equal_is_exact():
    # Every weight already has the same size: binary loses nothing.
    w = [0.5, -0.5, 0.5, -0.5]
    return all_close(binary_scaled(w), w)


def test_scaled_all_zeros_stay_zero():
    return all_close(binary_scaled([0, 0, 0]), [0, 0, 0])


def test_scaled_all_negative():
    return all_close(binary_scaled([-1, -2, -3]), [-2, -2, -2])


def test_scaled_keeps_every_sign():
    w = [0.01, -0.02, 0.3, -0.7, 0.05]
    return all(sign(a) == sign(b) for a, b in zip(w, binary_scaled(w)))


def test_scale_fixes_the_loud_choir():
    # The problem today's idea fixes: small weights become far too loud as +-1.
    w = [0.03, -0.01, 0.02, -0.04]
    plain_err = mean_absolute_error(w, binary_plain(w))
    scaled_err = mean_absolute_error(w, binary_scaled(w))
    return scaled_err < plain_err / 10


# --- Day 9's neuron ---

def test_neuron_real_answer():
    return close(neuron_answer([0.21, -0.48, 0.73], [1, 2, 3]), 1.44)


def test_neuron_plain_answer():
    # 1 - 2 + 3 = 2: only adding and subtracting, no multiply.
    return close(neuron_answer(binary_plain([0.21, -0.48, 0.73]), [1, 2, 3]), 2)


def test_neuron_scaled_closer_than_plain():
    w, x = [0.21, -0.48, 0.73], [1, 2, 3]
    real = neuron_answer(w, x)
    plain_miss = abs(neuron_answer(binary_plain(w), x) - real)
    scaled_miss = abs(neuron_answer(binary_scaled(w), x) - real)
    return close(scaled_miss, 0.4933) and scaled_miss < plain_miss


# --- Binary vs 2-bit linear, on a crowd near 0 with one outlier ---

def test_two_bit_linear_crushes_the_crowd():
    # Like the real q_lin grid: the outlier stretches 2-bit linear's buttons,
    # so the whole crowd snaps to 0.
    w = [0.01, -0.03, 0.02, -0.02, 0.03, -0.01, 0.7]
    out = squeeze(w, 2)
    return all(close(o, 0, 0.001) for o in out[:-1])


def test_binary_beats_two_bit_on_a_crowd():
    # 1 bit beats 2 bits here: binary's two buttons sit inside the crowd.
    # Like the real grid, the crowd is huge (120) and the outlier is rare (1).
    w = [0.01, -0.03, 0.02, -0.02, 0.03, -0.01] * 20 + [0.7]
    return mean_absolute_error(w, binary_scaled(w)) < mean_absolute_error(w, squeeze(w, 2))


def test_two_bit_wins_when_outliers_are_common():
    # Honest check: 1 outlier in 10 drags binary's average size up too far.
    w = [0.01, -0.03, 0.02, -0.02, 0.03, -0.01, 0.01, -0.02, 0.02, 0.7]
    return mean_absolute_error(w, squeeze(w, 2)) < mean_absolute_error(w, binary_scaled(w))


def test_two_bit_wins_on_an_even_spread():
    # Honest check: when weights spread evenly, 2 bits beats 1 bit.
    w = [-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9] * 3
    return mean_absolute_error(w, squeeze(w, 2)) < mean_absolute_error(w, binary_scaled(w))


if __name__ == "__main__":
    tests = [
        test_sign_positive,
        test_sign_negative,
        test_sign_zero_goes_to_plus_one,
        test_plain_toy,
        test_plain_only_two_values,
        test_plain_all_zeros,
        test_average_size_toy,
        test_average_size_ignores_signs,
        test_average_size_all_zeros,
        test_average_size_empty,
        test_scaled_toy,
        test_scaled_only_two_values,
        test_scaled_all_equal_is_exact,
        test_scaled_all_zeros_stay_zero,
        test_scaled_all_negative,
        test_scaled_keeps_every_sign,
        test_scale_fixes_the_loud_choir,
        test_neuron_real_answer,
        test_neuron_plain_answer,
        test_neuron_scaled_closer_than_plain,
        test_two_bit_linear_crushes_the_crowd,
        test_binary_beats_two_bit_on_a_crowd,
        test_two_bit_wins_when_outliers_are_common,
        test_two_bit_wins_on_an_even_spread,
    ]
    passed = 0
    for t in tests:
        ok = t()
        print(("PASS  " if ok else "FAIL  ") + t.__name__)
        passed += ok
    print(f"\n{passed} of {len(tests)} tests passed")
