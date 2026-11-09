# Checks for two_level.py. Run with:  python test_two_level.py
# These use made-up weights, so they run fast and need no download.

import random

import torch

from two_level import (squeeze_scales, squeeze_grid_two_level,
                       scale_bits_per_weight, squeeze_model_two_level)
from per_group import squeeze_per_group
from race import mean_absolute_error


def flatten(rows):
    return [w for row in rows for w in row]


def close(a, b, tol=1e-9):
    return len(a) == len(b) and all(abs(x - y) < tol for x, y in zip(a, b))


def crowd_rows(seed, n_rows=10, n_cols=192):
    # Rows shaped like a real layer: a crowd near 0, plus one outlier.
    rng = random.Random(seed)
    rows = [[rng.gauss(0, 0.05) for _ in range(n_cols)] for _ in range(n_rows)]
    rows[3][7] = 0.9
    return rows


# --- Our hand-worked example, exactly ---

def test_hand_example_scale_buttons():
    # Scales 0.30 and 0.04, buttons 0..10  ->  grid scale 0.03, buttons [10, 1].
    buttons, grid_scale = squeeze_scales([0.30, 0.04], 10)
    return buttons == [10, 1] and abs(grid_scale - 0.03) < 1e-12


def test_hand_example_b_scale_shrinks():
    # B's scale comes back as 1 x 0.03 = 0.03, not 0.04.
    buttons, grid_scale = squeeze_scales([0.30, 0.04], 10)
    return abs(buttons[1] * grid_scale - 0.03) < 1e-12


def test_hand_example_weights():
    # Same two groups as by hand, 2-bit weights. To get buttons 0..10 for the
    # scales, we copy the four steps of two_level.py using squeeze_scales directly.
    from zero_point import get_start_and_step_with_zero, get_zero_point, quantize, dequantize
    group_b = [0.00, 0.05, 0.08, 0.12]
    buttons, grid_scale = squeeze_scales([0.30, 0.04], 10)
    step = buttons[1] * grid_scale
    start, _ = get_start_and_step_with_zero(group_b, 2)
    zp = get_zero_point(start, step)
    back = dequantize(quantize(group_b, zp, step, 2), zp, step)
    # The last weight wanted button 4, got clipped to 3 -> 0.09 instead of 0.12.
    return close(back, [0.00, 0.06, 0.09, 0.09], tol=1e-9)


# --- squeeze_scales by itself ---

def test_biggest_scale_is_exact():
    # The biggest scale always lands exactly on the top button.
    buttons, grid_scale = squeeze_scales([0.004, 0.001, 0.0153, 0.0025], 255)
    return buttons[2] == 255 and abs(255 * grid_scale - 0.0153) < 1e-15


def test_buttons_fit_in_8_bits():
    rng = random.Random(1)
    scales = [rng.uniform(0.0001, 0.02) for _ in range(1000)]
    buttons, _ = squeeze_scales(scales, 255)
    return all(1 <= b <= 255 for b in buttons)


def test_tiny_scale_never_zero():
    # A scale far smaller than one button must not become 0 (that would erase its group).
    buttons, _ = squeeze_scales([1.0, 0.0000001], 255)
    return buttons[1] == 1


def test_all_scales_equal():
    buttons, grid_scale = squeeze_scales([0.01, 0.01, 0.01], 255)
    return buttons == [255, 255, 255] and abs(buttons[0] * grid_scale - 0.01) < 1e-15


def test_more_scale_bits_less_scale_error():
    rng = random.Random(2)
    scales = [rng.uniform(0.0005, 0.02) for _ in range(500)]
    errs = []
    for top in [15, 255, 65535]:            # 4, 8 and 16-bit scale buttons
        buttons, g = squeeze_scales(scales, top)
        errs.append(mean_absolute_error(scales, [b * g for b in buttons]))
    return errs[0] > errs[1] > errs[2]


# --- Whole grid ---

def test_same_shape():
    rows = [[0.01 * i for i in range(50)] for _ in range(3)]      # 50 doesn't divide by 16
    out = squeeze_grid_two_level(rows, 4)
    return len(out) == 3 and all(len(r) == 50 for r in out)


def test_close_to_day15_at_8_bit_scales():
    # 8-bit scales should add only a little error on top of Day 15.
    rows = crowd_rows(3)
    flat = flatten(rows)
    for bits in [8, 4, 2]:
        day15 = mean_absolute_error(flat, flatten(squeeze_per_group(rows, bits)))
        today = mean_absolute_error(flat, flatten(squeeze_grid_two_level(rows, bits)))
        if today > 2 * day15:
            return False
    return True


def test_never_better_than_day15_by_much():
    # Squeezing the scales loses information, so it shouldn't beat Day 15 clearly.
    rows = crowd_rows(4)
    flat = flatten(rows)
    day15 = mean_absolute_error(flat, flatten(squeeze_per_group(rows, 4)))
    today = mean_absolute_error(flat, flatten(squeeze_grid_two_level(rows, 4)))
    return today >= 0.95 * day15


def test_huge_scale_bits_match_day15():
    # With 32-bit scale buttons, rounding the scale changes almost nothing.
    rows = crowd_rows(5)
    a = flatten(squeeze_per_group(rows, 4))
    b = flatten(squeeze_grid_two_level(rows, 4, scale_bits=32))
    return close(a, b, tol=1e-6)


def test_fewer_scale_bits_more_error():
    rows = crowd_rows(6)
    flat = flatten(rows)
    errs = [mean_absolute_error(flat, flatten(squeeze_grid_two_level(rows, 4, scale_bits=s)))
            for s in [2, 4, 8]]
    return errs[0] > errs[1] > errs[2]


def test_more_weight_bits_less_error():
    rows = crowd_rows(7)
    flat = flatten(rows)
    errs = [mean_absolute_error(flat, flatten(squeeze_grid_two_level(rows, b)))
            for b in [2, 4, 8]]
    return errs[0] > errs[1] > errs[2]


def test_all_zero_grid():
    rows = [[0.0] * 32 for _ in range(2)]
    return squeeze_grid_two_level(rows, 4) == rows


def test_negatives_only():
    rows = [[-0.4, -0.2, -0.1, -0.3] * 8 for _ in range(2)]
    return mean_absolute_error(flatten(rows), flatten(squeeze_grid_two_level(rows, 8))) < 0.01


def test_each_group_few_values():
    # At 2 bits, each group of 16 may still use at most 4 different values.
    rows = crowd_rows(8)
    out = squeeze_grid_two_level(rows, 2)
    for row in out:
        for start in range(0, len(row), 16):
            if len(set(row[start:start + 16])) > 4:
                return False
    return True


# --- The bill ---

def test_scale_bill():
    return (scale_bits_per_weight() == 0.5
            and scale_bits_per_weight(16, 32) == 2
            and scale_bits_per_weight(32, 8) == 0.25)


def test_total_bill_4_bit():
    # 4 bits per weight + 0.5 for the scale = 4.5, down from Day 15's 6.
    return 4 + scale_bits_per_weight() == 4.5 and 4 + 32 / 16 == 6


# --- Whole (tiny) model ---

def tiny_model():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(32, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))


def test_original_untouched():
    model = tiny_model()
    before = model[0].weight.data.clone()
    squeeze_model_two_level(model, 2)
    return torch.equal(before, model[0].weight.data)


def test_counts_every_grid():
    _, grids = squeeze_model_two_level(tiny_model(), 4)
    return grids == 2


def test_biases_untouched():
    model = tiny_model()
    squeezed, _ = squeeze_model_two_level(model, 2)
    return torch.equal(model[0].bias.data, squeezed[0].bias.data)


def test_model_rows_match_function():
    model = tiny_model()
    squeezed, _ = squeeze_model_two_level(model, 4)
    expected = squeeze_grid_two_level(model[0].weight.data.tolist(), 4)
    got = squeezed[0].weight.data.tolist()
    return all(close(a, b, tol=1e-6) for a, b in zip(expected, got))


if __name__ == "__main__":
    tests = [
        test_hand_example_scale_buttons,
        test_hand_example_b_scale_shrinks,
        test_hand_example_weights,
        test_biggest_scale_is_exact,
        test_buttons_fit_in_8_bits,
        test_tiny_scale_never_zero,
        test_all_scales_equal,
        test_more_scale_bits_less_scale_error,
        test_same_shape,
        test_close_to_day15_at_8_bit_scales,
        test_never_better_than_day15_by_much,
        test_huge_scale_bits_match_day15,
        test_fewer_scale_bits_more_error,
        test_more_weight_bits_less_error,
        test_all_zero_grid,
        test_negatives_only,
        test_each_group_few_values,
        test_scale_bill,
        test_total_bill_4_bit,
        test_original_untouched,
        test_counts_every_grid,
        test_biases_untouched,
        test_model_rows_match_function,
    ]
    passed = 0
    for t in tests:
        ok = t()
        print(("PASS  " if ok else "FAIL  ") + t.__name__)
        passed += ok
    print(f"\n{passed} of {len(tests)} tests passed")
