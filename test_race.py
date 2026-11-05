# Checks for race.py. Run with:  python test_race.py
# These use tiny toy lists, so they run fast and need no download.

import random

from race import mean_absolute_error, linear_buttons, kmeans_squeeze, race
from look_inside import squeeze
from kmeans_quantize import kmeans


def close(a, b):
    # Decimals are never exact on a computer, so "equal" means "very close".
    return abs(a - b) < 1e-9


def bell_weights(count, seed):
    # Made-up weights shaped like a real layer: a big crowd near 0, a few far out.
    rng = random.Random(seed)
    return [rng.gauss(0, 0.05) for _ in range(count)] + [0.7, -0.6]


# --- mean_absolute_error ---

def test_mae_signs_do_not_cancel():
    # +0.1 and -0.1 would average to 0. MAE must say 0.10.
    return close(mean_absolute_error([0.3, 0.5], [0.4, 0.4]), 0.1)


def test_mae_perfect_copy_is_zero():
    return close(mean_absolute_error([1.0, -2.0, 3.0], [1.0, -2.0, 3.0]), 0)


def test_mae_toy():
    # misses 1, 2, 3  ->  average 2
    return close(mean_absolute_error([0, 0, 0], [1, -2, 3]), 2)


# --- linear_buttons ---

def test_linear_buttons_toy():
    # -2..4 at 2 bits: step 2, zero point 1  ->  [-2, 0, 2, 4]
    buttons = linear_buttons([-2, 1, 4], 2)
    return all(close(a, b) for a, b in zip(buttons, [-2, 0, 2, 4])) and len(buttons) == 4


def test_linear_buttons_count():
    w = bell_weights(100, seed=1)
    return all(len(linear_buttons(w, b)) == 2 ** b for b in [2, 4, 8])


def test_linear_buttons_include_zero():
    # Day 6/7 promise: real 0 always has a button.
    w = bell_weights(100, seed=2)
    return all(any(close(x, 0) for x in linear_buttons(w, b)) for b in [2, 4, 8])


# --- kmeans_squeeze ---

def test_kmeans_matches_day4_toy():
    # scikit-learn must agree with our own Day 4 carts on the toy weights.
    w = [0.31, -0.82, 0.07, -0.45, 0.66, 0.31, 0.07]  # 5 different values, 2 bits = 4 carts
    ours, groups = kmeans(w, linear_buttons(w, 2))
    ours_squeezed = [ours[g] for g in groups]
    theirs = kmeans_squeeze(w, 2)
    return all(abs(a - b) < 1e-6 for a, b in zip(ours_squeezed, theirs))


def test_kmeans_uses_at_most_2_to_the_bits_values():
    w = bell_weights(500, seed=3)
    return all(len(set(kmeans_squeeze(w, b))) <= 2 ** b for b in [2, 4])


def test_kmeans_few_weights_kept_exact():
    # 3 weights, 4 carts: every weight gets its own cart.
    w = [0.1, -0.2, 0.3]
    return kmeans_squeeze(w, 2) == w


def test_kmeans_all_equal():
    # Same trap as Day 5: every weight is 5. Nothing to squeeze, no error.
    return kmeans_squeeze([5.0] * 50, 2) == [5.0] * 50


def test_kmeans_all_zero():
    return kmeans_squeeze([0.0] * 50, 4) == [0.0] * 50


def test_kmeans_negatives_only():
    w = [-x / 100 for x in range(1, 200)]
    return mean_absolute_error(w, kmeans_squeeze(w, 2)) < mean_absolute_error(w, squeeze(w, 2))


# --- the race (the problem today's idea fixes) ---

def test_kmeans_beats_linear_on_bell_shape():
    # A crowd near 0 with far-out edges: linear wastes buttons, K-means doesn't.
    w = bell_weights(2000, seed=4)
    scores = race(w, bits_list=(2, 4, 8))
    return all(km < lin for lin, km in scores.values())


def test_more_bits_less_error_for_both():
    w = bell_weights(2000, seed=5)
    s = race(w, bits_list=(2, 4, 8))
    return s[8][0] < s[4][0] < s[2][0] and s[8][1] < s[4][1] < s[2][1]


def test_even_weights_close_race():
    # Evenly spread weights: linear's even buttons are already in the right place.
    # K-means can't do much better, so the two should be close (within 10%).
    w = [x / 1000 for x in range(-1000, 1001)]
    lin, km = race(w, bits_list=(4,))[4]
    return km <= lin * 1.1 and km >= lin * 0.9


if __name__ == "__main__":
    tests = [
        test_mae_signs_do_not_cancel,
        test_mae_perfect_copy_is_zero,
        test_mae_toy,
        test_linear_buttons_toy,
        test_linear_buttons_count,
        test_linear_buttons_include_zero,
        test_kmeans_matches_day4_toy,
        test_kmeans_uses_at_most_2_to_the_bits_values,
        test_kmeans_few_weights_kept_exact,
        test_kmeans_all_equal,
        test_kmeans_all_zero,
        test_kmeans_negatives_only,
        test_kmeans_beats_linear_on_bell_shape,
        test_more_bits_less_error_for_both,
        test_even_weights_close_race,
    ]
    passed = 0
    for t in tests:
        ok = t()
        print(("PASS  " if ok else "FAIL  ") + t.__name__)
        passed += ok
    print(f"\n{passed} of {len(tests)} tests passed")
