# Checks for per_group.py. Run with:  python test_per_group.py
# These use made-up weights, so they run fast and need no download.

import random

import torch

from per_group import (cut_into_groups, squeeze_row_per_group, squeeze_per_group,
                       extra_bits_per_weight, squeeze_model_per_group)
from per_channel import squeeze_per_channel
from look_inside import squeeze
from race import mean_absolute_error


def flatten(rows):
    return [w for row in rows for w in row]


def close(a, b):
    return len(a) == len(b) and all(abs(x - y) < 1e-9 for x, y in zip(a, b))


def outlier_rows(seed):
    # Rows shaped like a real layer: a crowd near 0, plus one big outlier in row 3.
    rng = random.Random(seed)
    rows = [[rng.gauss(0, 0.05) for _ in range(192)] for _ in range(10)]
    rows[3][7] = 0.9
    return rows


# --- Cutting a row into groups ---

def test_cut_even():
    return cut_into_groups([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]


def test_cut_uneven():
    # The toy example from the code: the last group is shorter.
    return cut_into_groups([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]


def test_cut_group_bigger_than_row():
    return cut_into_groups([1, 2, 3], 16) == [[1, 2, 3]]


def test_cut_keeps_every_weight():
    row = list(range(100))
    return flatten(cut_into_groups(row, 16)) == row


# --- Squeezing one row ---

def test_toy_outlier_group():
    # The example from the code: per-channel wipes out the small weights,
    # per-group keeps them.
    row = [0.1, 0.2, 0.3, -5, 0, 5]
    per_channel = squeeze(row, 2)
    per_group = squeeze_row_per_group(row, 2, group_size=3)
    return per_channel[:3] == [0.0, 0.0, 0.0] and close(per_group[:3], [0.1, 0.2, 0.3])


def test_same_length():
    row = [0.01 * i for i in range(50)]           # 50 doesn't divide by 16
    return len(squeeze_row_per_group(row, 4)) == 50


def test_group_as_big_as_row_is_per_channel():
    # One group per row = per-channel. Same answer.
    rows = outlier_rows(1)
    return all(close(a, b) for a, b in zip(squeeze_per_group(rows, 4, group_size=192),
                                           squeeze_per_channel(rows, 4)))


def test_groups_dont_affect_each_other():
    # Changing group 2 must not change how group 1 is squeezed.
    g1 = [0.1, -0.05, 0.2, 0.0]
    a = squeeze_row_per_group(g1 + [0.3, 0.1, -0.2, 0.0], 4, group_size=4)[:4]
    b = squeeze_row_per_group(g1 + [90, -70, 50, 0.0], 4, group_size=4)[:4]
    return close(a, b)


def test_each_group_has_few_values():
    # At 2 bits, each group may use at most 4 different values.
    rng = random.Random(2)
    row = [rng.gauss(0, 1) for _ in range(160)]
    out = squeeze_row_per_group(row, 2, group_size=16)
    return all(len(set(g)) <= 4 for g in cut_into_groups(out, 16))


def test_all_equal_group():
    # A group where every weight is the same. Must not crash (divide by zero trap).
    out = squeeze_row_per_group([0.5] * 16 + [0.1, 0.2, 0.3], 4)
    return len(out) == 19


def test_all_zero_group():
    # A group of zeros stays all zeros.
    out = squeeze_row_per_group([0.0] * 16 + [0.1, -0.2, 0.3], 4)
    return out[:16] == [0.0] * 16


def test_negatives_only():
    row = [-0.4, -0.2, -0.1, -0.3] * 8
    return mean_absolute_error(row, squeeze_row_per_group(row, 8)) < 0.01


# --- The problem today fixes ---

def test_beats_per_channel_on_outlier_grid():
    # Per-group should win at every bit width.
    rows = outlier_rows(3)
    flat = flatten(rows)
    for bits in [8, 4, 2]:
        channel_err = mean_absolute_error(flat, flatten(squeeze_per_channel(rows, bits)))
        group_err = mean_absolute_error(flat, flatten(squeeze_per_group(rows, bits)))
        if group_err >= channel_err:
            return False
    return True


def test_outlier_row_rescued():
    # In row 3 (the one with the outlier), per-channel rounds many weights to 0.
    # Per-group should keep far more of them alive.
    rows = outlier_rows(4)
    channel_zeros = squeeze_per_channel(rows, 4)[3].count(0.0)
    group_zeros = squeeze_per_group(rows, 4)[3].count(0.0)
    return group_zeros < channel_zeros / 2


def test_outlier_only_hurts_its_group():
    # Only the outlier's own group (weights 0..15 of row 3) should be blurry.
    # Every other group in row 3 should be as good as in a row with no outlier.
    rows = outlier_rows(5)
    row3 = rows[3]
    out = squeeze_row_per_group(row3, 4)
    bad_group = mean_absolute_error(row3[:16], out[:16])
    other_groups = mean_absolute_error(row3[16:], out[16:])
    return bad_group > 3 * other_groups


def test_smaller_groups_less_error():
    rows = outlier_rows(6)
    flat = flatten(rows)
    errs = [mean_absolute_error(flat, flatten(squeeze_per_group(rows, 4, group_size=g)))
            for g in [192, 64, 16]]
    return errs[0] > errs[1] > errs[2]


def test_more_bits_less_error():
    rows = outlier_rows(7)
    flat = flatten(rows)
    errs = [mean_absolute_error(flat, flatten(squeeze_per_group(rows, b))) for b in [2, 4, 8]]
    return errs[0] > errs[1] > errs[2]


# --- The cost (the dinner bill) ---

def test_extra_bits():
    return (extra_bits_per_weight(16) == 2
            and extra_bits_per_weight(32) == 1
            and extra_bits_per_weight(768) == 32 / 768
            and extra_bits_per_weight(1) == 32)


def test_group_of_one_costs_more_than_float32():
    # The joke from today: groups of 1 cost 4 + 32 = 36 bits per weight.
    return 4 + extra_bits_per_weight(1) > 32


# --- Whole (tiny) model ---

def tiny_model():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(32, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))


def test_original_untouched():
    model = tiny_model()
    before = model[0].weight.data.clone()
    squeeze_model_per_group(model, 2)
    return torch.equal(before, model[0].weight.data)


def test_counts_every_grid():
    _, grids = squeeze_model_per_group(tiny_model(), 4)
    return grids == 2


def test_shapes_kept():
    model = tiny_model()
    squeezed, _ = squeeze_model_per_group(model, 4)
    return all(a.weight.shape == b.weight.shape
               for a, b in zip([model[0], model[2]], [squeezed[0], squeezed[2]]))


def test_biases_untouched():
    model = tiny_model()
    squeezed, _ = squeeze_model_per_group(model, 2)
    return torch.equal(model[0].bias.data, squeezed[0].bias.data)


def test_model_rows_match_function():
    # Each row inside the model is squeezed exactly like squeeze_per_group does it.
    # The model stores float32 (about 7 digits), so we compare to 6 digits.
    model = tiny_model()
    squeezed, _ = squeeze_model_per_group(model, 3)
    expected = squeeze_per_group(model[0].weight.data.tolist(), 3)
    got = squeezed[0].weight.data.tolist()
    return all(abs(e[i] - g[i]) < 1e-6 for e, g in zip(expected, got) for i in range(len(e)))


if __name__ == "__main__":
    tests = [
        test_cut_even,
        test_cut_uneven,
        test_cut_group_bigger_than_row,
        test_cut_keeps_every_weight,
        test_toy_outlier_group,
        test_same_length,
        test_group_as_big_as_row_is_per_channel,
        test_groups_dont_affect_each_other,
        test_each_group_has_few_values,
        test_all_equal_group,
        test_all_zero_group,
        test_negatives_only,
        test_beats_per_channel_on_outlier_grid,
        test_outlier_row_rescued,
        test_outlier_only_hurts_its_group,
        test_smaller_groups_less_error,
        test_more_bits_less_error,
        test_extra_bits,
        test_group_of_one_costs_more_than_float32,
        test_original_untouched,
        test_counts_every_grid,
        test_shapes_kept,
        test_biases_untouched,
        test_model_rows_match_function,
    ]
    passed = 0
    for t in tests:
        ok = t()
        print(("PASS  " if ok else "FAIL  ") + t.__name__)
        passed += ok
    print(f"\n{passed} of {len(tests)} tests passed")
