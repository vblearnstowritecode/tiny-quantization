# Checks for fp4.py. Run with:  python test_fp4.py
# No model download, so it runs in a few seconds.

from fp4 import (
    FP4_BUTTONS,
    nearest_button,
    fp4_scale,
    squeeze_fp4,
    crowd_buttons,
)
from look_inside import squeeze
from race import mean_absolute_error


def close(a, b, tol=0.0001):
    return abs(a - b) < tol


# --- The buttons ---

def test_15_buttons():
    # 8 positive values + 7 negative mirrors (0 is shared) = 15.
    return len(FP4_BUTTONS) == 15


def test_buttons_are_symmetric():
    return all(-b in FP4_BUTTONS for b in FP4_BUTTONS)


def test_gaps_grow_far_out():
    # Close near 0 (0.5), wide far out (2).
    return FP4_BUTTONS[8] - FP4_BUTTONS[7] == 0.5 and FP4_BUTTONS[14] - FP4_BUTTONS[13] == 2


# --- Snapping to the nearest button ---

def test_nearest_normal():
    return nearest_button(2.004) == 2 and nearest_button(4.5) == 4


def test_nearest_on_a_button():
    return all(nearest_button(b) == b for b in FP4_BUTTONS)


def test_nearest_negative():
    return nearest_button(-1.2) == -1 and nearest_button(-5.4) == -6


def test_nearest_beyond_6_clamps():
    return nearest_button(9) == 6 and nearest_button(-100) == -6


def test_nearest_tie_keeps_first_met():
    # 2.5 is halfway between 2 and 3: we meet 2 first (buttons go low to high).
    # -2.5 is halfway between -3 and -2: we meet -3 first.
    return nearest_button(2.5) == 2 and nearest_button(-2.5) == -3


# --- The scale ---

def test_scale_bus_stop_toy():
    return fp4_scale([100, 200.4, 450, 600]) == 100


def test_scale_negative_biggest():
    # The biggest weight ignoring sign is -12, so scale = 12 / 6 = 2.
    return fp4_scale([1, -12, 5]) == 2


def test_scale_all_zeros():
    return fp4_scale([0, 0, 0]) == 1


# --- The full round trip ---

def test_squeeze_bus_stop_toy():
    out = squeeze_fp4([100, 200.4, 450, 600])
    return all(close(a, b) for a, b in zip(out, [100, 200, 400, 600]))


def test_150_walks_0_and_500_walks_100():
    out = squeeze_fp4([150, 500, 600])
    return close(out[0], 150) and close(abs(500 - out[1]), 100)


def test_all_equal_weights_exact():
    # Every weight is the biggest, so every weight lands on button 6.
    out = squeeze_fp4([0.3, 0.3, 0.3])
    return all(close(o, 0.3) for o in out)


def test_all_zeros_stay_zero():
    return squeeze_fp4([0, 0, 0]) == [0, 0, 0]


def test_zero_stays_exact():
    out = squeeze_fp4([0, 0.37, -0.81, 0, 0.05])
    return out[0] == 0 and out[3] == 0


def test_error_at_most_half_the_local_gap():
    # Each weight walks at most half the gap between its two nearest stops.
    weights = [i / 100 - 6 for i in range(1201)]   # -6.00 .. 6.00, scale = 1
    out = squeeze_fp4(weights)
    for w, s in zip(weights, out):
        below = max(b for b in FP4_BUTTONS if b <= w)
        above = min(b for b in FP4_BUTTONS if b >= w)
        if abs(w - s) > (above - below) / 2 + 1e-9:
            return False
    return True


def test_downtown_walks_shorter_than_suburbs():
    # The idea: near 0 the worst walk is 0.25, far out it is 1.
    near = [i / 100 for i in range(0, 201)]     # 0 .. 2   (downtown)
    far = [i / 100 for i in range(400, 601)]    # 4 .. 6   (suburbs)
    worst_near = max(abs(w - nearest_button(w)) for w in near)
    worst_far = max(abs(w - nearest_button(w)) for w in far)
    return close(worst_near, 0.25) and close(worst_far, 1)


def test_fp4_beats_linear_on_a_crowd():
    # The problem today fixes: a crowd near 0 plus one far-out weight.
    # Even buttons waste most stops in the empty suburbs. FP4 keeps them downtown.
    crowd = [((i * 37) % 21 - 10) / 100 for i in range(200)]   # -0.10 .. 0.10
    weights = crowd + [-0.7, 0.5]
    linear_mae = mean_absolute_error(weights, squeeze(weights, 4))
    fp4_mae = mean_absolute_error(weights, squeeze_fp4(weights))
    return fp4_mae < linear_mae


def test_crowd_buttons_toy():
    # Near 0: 0, 0.05 and -0.05 are 3 different buttons. 0.5 is outside the crowd.
    return crowd_buttons([0, 0.05, 0.05, -0.05, 0.5]) == 3


def test_crowd_buttons_empty():
    return crowd_buttons([0.5, -0.9]) == 0


if __name__ == "__main__":
    tests = [
        test_15_buttons,
        test_buttons_are_symmetric,
        test_gaps_grow_far_out,
        test_nearest_normal,
        test_nearest_on_a_button,
        test_nearest_negative,
        test_nearest_beyond_6_clamps,
        test_nearest_tie_keeps_first_met,
        test_scale_bus_stop_toy,
        test_scale_negative_biggest,
        test_scale_all_zeros,
        test_squeeze_bus_stop_toy,
        test_150_walks_0_and_500_walks_100,
        test_all_equal_weights_exact,
        test_all_zeros_stay_zero,
        test_zero_stays_exact,
        test_error_at_most_half_the_local_gap,
        test_downtown_walks_shorter_than_suburbs,
        test_fp4_beats_linear_on_a_crowd,
        test_crowd_buttons_toy,
        test_crowd_buttons_empty,
    ]
    passed = 0
    for t in tests:
        ok = t()
        print(("PASS  " if ok else "FAIL  ") + t.__name__)
        passed += ok
    print(f"\n{passed} of {len(tests)} tests passed")
