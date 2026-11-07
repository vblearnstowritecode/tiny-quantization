# Checks for per_channel.py. Run with:  python test_per_channel.py
# These use made-up weights, so they run fast and need no download.

import random

import torch

from per_channel import squeeze_per_channel, squeeze_model_per_channel
from look_inside import squeeze
from race import mean_absolute_error


def flatten(rows):
    return [w for row in rows for w in row]


def close(a, b):
    return all(abs(x - y) < 1e-9 for x, y in zip(a, b))


# --- One grid ---

def test_toy_outlier_row():
    # The example from the code: per-tensor wipes out row 1, per-channel keeps it.
    rows = [[0.1, 0.2, 0.3], [-5, 0, 5]]
    per_tensor = squeeze(flatten(rows), 2)
    per_channel = squeeze_per_channel(rows, 2)
    return per_tensor[:3] == [0.0, 0.0, 0.0] and close(per_channel[0], [0.1, 0.2, 0.3])


def test_same_shape():
    # Same number of rows, same length per row.
    rows = [[0.1, -0.2, 0.3, 0.4], [1, 2, 3, 4], [0, 0, 0, 0]]
    out = squeeze_per_channel(rows, 4)
    return len(out) == 3 and all(len(r) == 4 for r in out)


def test_one_row_same_as_per_tensor():
    # With only one row, per-channel and per-tensor are the same thing.
    row = [0.31, -0.07, 0.45, -0.52, 0.0, 0.18]
    return close(squeeze_per_channel([row], 3)[0], squeeze(row, 3))


def test_rows_dont_affect_each_other():
    # Changing row 2 must not change how row 1 is squeezed.
    row1 = [0.1, -0.05, 0.2, 0.0]
    a = squeeze_per_channel([row1, [0.3, 0.1, -0.2, 0.0]], 4)[0]
    b = squeeze_per_channel([row1, [90, -70, 50, 0.0]], 4)[0]
    return close(a, b)


def test_each_row_has_few_values():
    # At 2 bits, each row may use at most 4 different values.
    rng = random.Random(1)
    rows = [[rng.gauss(0, 1) for _ in range(100)] for _ in range(5)]
    out = squeeze_per_channel(rows, 2)
    return all(len(set(r)) <= 4 for r in out)


def test_all_equal_row():
    # A row where every weight is the same. Must not crash (divide by zero trap).
    out = squeeze_per_channel([[0.5, 0.5, 0.5], [0.1, 0.2, 0.3]], 4)
    return len(out[0]) == 3


def test_all_zero_row():
    # A row of zeros stays all zeros.
    out = squeeze_per_channel([[0, 0, 0], [0.1, -0.2, 0.3]], 4)
    return out[0] == [0, 0, 0]


def test_negatives_only_row():
    # A row with only negative weights still gets sensible values back.
    row = [-0.4, -0.2, -0.1, -0.3]
    out = squeeze_per_channel([row], 8)[0]
    return mean_absolute_error(row, out) < 0.01


def test_never_worse_on_outlier_grid():
    # Grid shaped like a real layer: every row is a crowd near 0,
    # but one row holds a big outlier. Per-channel should win at every bit width.
    rng = random.Random(2)
    rows = [[rng.gauss(0, 0.05) for _ in range(200)] for _ in range(20)]
    rows[3][7] = 0.9
    flat = flatten(rows)
    for bits in [8, 4, 2]:
        tensor_err = mean_absolute_error(flat, squeeze(flat, bits))
        channel_err = mean_absolute_error(flat, flatten(squeeze_per_channel(rows, bits)))
        if channel_err >= tensor_err:
            return False
    return True


def test_more_bits_less_error():
    rng = random.Random(3)
    rows = [[rng.gauss(0, 0.05) for _ in range(200)] for _ in range(10)]
    flat = flatten(rows)
    errs = [mean_absolute_error(flat, flatten(squeeze_per_channel(rows, b))) for b in [2, 4, 8]]
    return errs[0] > errs[1] > errs[2]


def test_fewer_vanish():
    # The problem today fixes: per-tensor rounds most of the crowd to 0.
    # Per-channel should keep far more weights alive (non-zero).
    rng = random.Random(4)
    rows = [[rng.gauss(0, 0.05) for _ in range(200)] for _ in range(20)]
    rows[3][7] = 0.9
    flat = flatten(rows)
    tensor_zeros = sum(1 for w in squeeze(flat, 4) if w == 0)
    channel_zeros = sum(1 for w in flatten(squeeze_per_channel(rows, 4)) if w == 0)
    return channel_zeros < tensor_zeros / 2


# --- Whole (tiny) model ---

def tiny_model():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(6, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))


def test_original_untouched():
    model = tiny_model()
    before = model[0].weight.data.clone()
    squeeze_model_per_channel(model, 2)
    return torch.equal(before, model[0].weight.data)


def test_counts_every_grid():
    _, grids = squeeze_model_per_channel(tiny_model(), 4)
    return grids == 2


def test_shapes_kept():
    model = tiny_model()
    squeezed, _ = squeeze_model_per_channel(model, 4)
    return all(a.weight.shape == b.weight.shape
               for a, b in zip([model[0], model[2]], [squeezed[0], squeezed[2]]))


def test_biases_untouched():
    model = tiny_model()
    squeezed, _ = squeeze_model_per_channel(model, 2)
    return torch.equal(model[0].bias.data, squeezed[0].bias.data)


def test_model_rows_match_function():
    # Each row inside the model is squeezed exactly like squeeze_per_channel does it.
    # The model stores float32 (about 7 digits), so we compare to 6 digits.
    model = tiny_model()
    squeezed, _ = squeeze_model_per_channel(model, 3)
    expected = squeeze_per_channel(model[0].weight.data.tolist(), 3)
    got = squeezed[0].weight.data.tolist()
    return all(abs(e[i] - g[i]) < 1e-6 for e, g in zip(expected, got) for i in range(len(e)))


if __name__ == "__main__":
    tests = [
        test_toy_outlier_row,
        test_same_shape,
        test_one_row_same_as_per_tensor,
        test_rows_dont_affect_each_other,
        test_each_row_has_few_values,
        test_all_equal_row,
        test_all_zero_row,
        test_negatives_only_row,
        test_never_worse_on_outlier_grid,
        test_more_bits_less_error,
        test_fewer_vanish,
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
