# Checks for activations.py. Run with:  python test_activations.py
# No model download, so it runs in a few seconds.

from activations import find_spike_channel, split_calm_and_spike, errors, stops_used
from look_inside import squeeze
from fp4 import squeeze_fp4
from race import mean_absolute_error


def close(a, b, tol=0.0001):
    return abs(a - b) < tol


# --- Finding the spiky judge ---

def test_spike_channel_toy():
    rows = [[1, -9, 2],
            [0, -8, 1],
            [20, -7, 1]]
    # Channel 0 has the single biggest number (20), but channel 1 is big for every token.
    return find_spike_channel(rows) == 1


def test_spike_channel_positive():
    return find_spike_channel([[0.1, 0.2, 5], [0.3, 0.1, 6]]) == 2


def test_spike_channel_all_equal():
    # No spike at all: we just keep the first channel.
    return find_spike_channel([[1, 1, 1], [1, 1, 1]]) == 0


def test_spike_channel_all_zero():
    return find_spike_channel([[0, 0], [0, 0]]) == 0


# --- Sorting groups into calm and spike piles ---

def test_split_toy():
    calm, spike = split_calm_and_spike([[1, -9, 2, 3]], spike_channel=1, group_size=2)
    return spike == [[1, -9]] and calm == [[2, 3]]


def test_split_one_spike_group_per_token():
    # 3 tokens, 32 channels, groups of 16: each token has 1 spike group and 1 calm group.
    rows = [[0.1] * 32 for _ in range(3)]
    calm, spike = split_calm_and_spike(rows, spike_channel=20, group_size=16)
    return len(spike) == 3 and len(calm) == 3


def test_split_keeps_every_number():
    rows = [list(range(48)), list(range(48, 96))]
    calm, spike = split_calm_and_spike(rows, spike_channel=5, group_size=16)
    everything = sorted(v for g in calm + spike for v in g)
    return everything == list(range(96))


# --- Counting stops ---

def test_stops_used_toy():
    return stops_used([0, 0, -2, -2, -6]) == 3


def test_stops_used_rounding():
    # 0.1 + 0.2 is 0.30000000000000004 in Python. It still counts as the same stop as 0.3.
    return stops_used([0.1 + 0.2, 0.3]) == 1


# --- Errors ---

def test_errors_all_zero_group():
    lin, fp4 = errors([[0] * 16])
    return lin == 0 and fp4 == 0


def test_errors_all_equal_group():
    lin, fp4 = errors([[0.7] * 16])
    return close(lin, 0) and close(fp4, 0)


# --- The ideas from today ---

ONE_SIDED = [-i / 10 for i in range(61)]       # -6.0 .. 0.0, nothing positive
TWO_SIDED = [i / 10 - 6 for i in range(121)]   # -6.0 .. 6.0


def test_fp4_wastes_its_plus_side_on_one_sided_group():
    # FP4 has 15 stops, but on a one-sided group only the 0 and - side get used: at most 8.
    # Linear fits min..max, so all 16 of its stops fall inside -6..0.
    return stops_used(squeeze_fp4(ONE_SIDED)) <= 8 and stops_used(squeeze(ONE_SIDED, 4)) == 16


def test_fp4_uses_both_sides_on_two_sided_group():
    return stops_used(squeeze_fp4(TWO_SIDED)) == 15


def test_linear_beats_fp4_on_one_sided_group():
    lin = mean_absolute_error(ONE_SIDED, squeeze(ONE_SIDED, 4))
    fp4 = mean_absolute_error(ONE_SIDED, squeeze_fp4(ONE_SIDED))
    return lin < fp4


def test_spike_snaps_small_numbers_to_zero():
    # The problem: 15 small numbers + one spike of -60. Every small number becomes 0.
    small = [0.3, -0.5, 0.2, 0.4, -0.1, 0.6, -0.3, 0.1, 0.5, -0.2, 0.3, -0.4, 0.2, 0.1, -0.6]
    group = small + [-60]
    lin = squeeze(group, 4)[:15]
    fp4 = squeeze_fp4(group)[:15]
    return all(v == 0 for v in lin) and all(v == 0 for v in fp4)


def test_without_spike_small_numbers_survive():
    small = [0.3, -0.5, 0.2, 0.4, -0.1, 0.6, -0.3, 0.1, 0.5, -0.2, 0.3, -0.4, 0.2, 0.1, -0.6]
    lin = squeeze(small, 4)
    fp4 = squeeze_fp4(small)
    return sum(1 for v in lin if v != 0) >= 13 and sum(1 for v in fp4 if v != 0) >= 13


def test_spike_group_error_bigger_than_calm_group():
    calm = [[0.3, -0.5, 0.2, 0.4, -0.1, 0.6, -0.3, 0.1]]
    spiky = [[0.3, -0.5, 0.2, 0.4, -0.1, 0.6, -0.3, -8.0]]
    calm_lin, calm_fp4 = errors(calm)
    spike_lin, spike_fp4 = errors(spiky)
    return spike_lin > calm_lin and spike_fp4 > calm_fp4


def test_more_bits_shrink_spike_damage():
    # Several bit widths: linear on a spike group gets better as bits go up.
    group = [0.3, -0.5, 0.2, 0.4, -0.1, 0.6, -0.3, -8.0]
    e2 = mean_absolute_error(group, squeeze(group, 2))
    e4 = mean_absolute_error(group, squeeze(group, 4))
    e8 = mean_absolute_error(group, squeeze(group, 8))
    return e8 < e4 < e2


# --- Run them all ---

if __name__ == "__main__":
    tests = [(name, fn) for name, fn in list(globals().items()) if name.startswith("test_")]
    passed = 0
    for name, fn in tests:
        ok = fn()
        print(("PASS  " if ok else "FAIL  ") + name)
        passed += ok
    print(f"\n{passed} of {len(tests)} tests passed")
