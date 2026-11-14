# Checks for clipping.py. Run with:  python test_clipping.py
# No model download, so it runs in a few seconds.

from clipping import (clip, clip_both_sides, clip_then_squeeze, clip_then_squeeze_fp4,
                      snapped_to_zero, best_cut_off)
from look_inside import squeeze
from fp4 import squeeze_fp4
from race import mean_absolute_error


def crowd_with_one_loud(size, loud=-5.0):
    # A quiet crowd spread evenly from -0.1 to +0.1, plus one loud value at the end.
    # Toy example, size 5:  [-0.1, -0.05, 0.0, 0.05, 0.1, -5.0]
    crowd = [-0.1 + 0.2 * i / (size - 1) for i in range(size)]
    return crowd + [loud]


# --- The fence (one side) ---

def test_clip_toy():
    return clip([-6, 0.2, -0.1, 1], -2) == [-2, 0.2, -0.1, 1]


def test_clip_leaves_values_above_alone():
    return clip([0.1, 0.5, 3], -2) == [0.1, 0.5, 3]


def test_clip_all_equal():
    return clip([-4, -4, -4], -1) == [-1, -1, -1]


def test_clip_all_zero():
    return clip([0, 0, 0], -1) == [0, 0, 0]


def test_clip_value_on_the_fence_stays():
    return clip([-2, -3], -2) == [-2, -2]


# --- Two fences (both sides) ---

def test_clip_both_toy():
    return clip_both_sides([-0.7, 0.05, 0.5], 0.2) == [-0.2, 0.05, 0.2]


def test_clip_both_wide_fence_changes_nothing():
    values = [-0.3, 0.1, 0.4]
    return clip_both_sides(values, 10) == values


def test_clip_both_negatives_only():
    return clip_both_sides([-5, -0.1], 1) == [-1, -0.1]


# --- Counting numbers that snap to 0 ---

def test_snapped_toy():
    # The 0.2 vanished. The real 0 was already 0, so it doesn't count.
    return snapped_to_zero([0.2, -0.1, 0, 3], [0, -0.1, 0, 3]) == 1


def test_snapped_none():
    return snapped_to_zero([1, 2], [1, 2]) == 0


# --- What clipping does to the stops ---

def test_clip_brings_quiet_values_closer():
    # After the loud value is clipped, the quiet values come back closer to the truth.
    values = crowd_with_one_loud(15)
    quiet = values[:-1]
    before = mean_absolute_error(quiet, squeeze(values, 4)[:-1])
    after = mean_absolute_error(quiet, clip_then_squeeze(values, -1, 4)[:-1])
    return after < before


def test_clip_hurts_the_loud_value():
    # The loud value pays: -5 now reads close to the fence, not -5.
    values = crowd_with_one_loud(15)
    return abs(clip_then_squeeze(values, -1, 4)[-1] - (-1)) < 0.1


def test_clip_fp4_brings_quiet_values_closer():
    values = crowd_with_one_loud(15)
    quiet = values[:-1]
    before = mean_absolute_error(quiet, squeeze_fp4(values)[:-1])
    after = mean_absolute_error(quiet, clip_then_squeeze_fp4(values, -1)[:-1])
    return after < before


# --- The Day 21 lesson: clipping pays only when the loud values are rare ---

def test_one_loud_in_16_clipping_loses():
    # Like the spike group: 1 loud value in 16. No cut-off beats "no clipping".
    values = crowd_with_one_loud(15)
    best, _ = best_cut_off(values, [-4, -3, -2, -1, -0.5])
    return best is None


def test_one_loud_in_1000_clipping_wins():
    # Like the weight grid: 1 loud value in 1000. Some cut-off beats "no clipping".
    values = crowd_with_one_loud(999)
    best, error = best_cut_off(values, [-4, -3, -2, -1, -0.5])
    no_clip = mean_absolute_error(values, squeeze(values, 4))
    return best is not None and error < no_clip


def test_too_tight_hurts_again():
    # The dial: a fence far inside the crowd is worse than a sensible fence.
    values = crowd_with_one_loud(999)
    sensible = mean_absolute_error(values, clip_then_squeeze(values, -0.5, 4))
    too_tight = mean_absolute_error(values, clip_then_squeeze(values, -0.01, 4))
    return too_tight > sensible


def test_rare_loud_wins_at_several_bit_widths():
    # At 2, 3 and 4 bits, clipping a rare loud value helps.
    values = crowd_with_one_loud(999)
    for bits in (2, 3, 4):
        best, _ = best_cut_off(values, [-2, -1, -0.5, -0.2], bits)
        if best is None:
            return False
    return True


def test_clipping_with_nothing_loud_changes_nothing():
    # All values above the fence: same answer as no clipping.
    values = [0.1, -0.2, 0.3, 0.05]
    return clip_then_squeeze(values, -1, 4) == squeeze(values, 4)


# --- Run them all ---

if __name__ == "__main__":
    tests = [(name, fn) for name, fn in list(globals().items()) if name.startswith("test_")]
    passed = 0
    for name, fn in tests:
        ok = fn()
        print(("PASS  " if ok else "FAIL  ") + name)
        passed += ok
    print(f"\n{passed} of {len(tests)} tests passed")
