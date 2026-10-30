# Checks for zero_point.py. Run with:  python test_zero_point.py
# Each test prints PASS or FAIL, so you can see exactly what works.

from shift_quantize import get_start_and_step
from zero_point import round_half_up, get_zero_point, quantize, dequantize


def round_trip(values, bits):
    # Quantize and convert back, the full Day 6 way.
    start, step = get_start_and_step(values, bits)
    zero_point = get_zero_point(start, step)
    slots = quantize(values, zero_point, step, bits)
    return slots, dequantize(slots, zero_point, step)


def test_round_half_up():
    # Halves go up, away from 0. Python's own round(0.5) would give 0.
    return (round_half_up(0.5) == 1 and round_half_up(1.5) == 2
            and round_half_up(2.4) == 2 and round_half_up(-0.5) == -1)


def test_zero_point_whole_steps():
    # Avi's example: start -6, step 2 -> 3 steps up to 0
    return get_zero_point(-6, 2) == 3


def test_zero_point_needs_a_nudge():
    # The README example: start -1, step 2 -> 0.5 steps -> rounds to slot 1
    return get_zero_point(-1, 2) == 1


def test_zero_point_when_start_is_zero():
    # Values 0..14 already have 0 on slot 0
    return get_zero_point(0, 2) == 0


def test_zero_comes_back_exactly():
    # The whole point of today: 0 in, 0 out. Day 5 gave back -1 here.
    slots, recovered = round_trip([-1, 0, 13], 3)
    return recovered[1] == 0


def test_many_zeros_all_exact():
    # ReLU-style outputs: lots of zeros, all must survive untouched
    values = [0, 0, 4, 0, 0, 7, 0, 0, -1, 13]
    slots, recovered = round_trip(values, 3)
    return all(r == 0 for v, r in zip(values, recovered) if v == 0)


def test_zero_exact_for_odd_ranges():
    # Messier ranges, different bit widths: 0 must still come back as exactly 0
    for values in ([-0.82, 0, 0.66], [-3.7, 0, 11.2], [-100, 0, 1]):
        for bits in (2, 3, 4, 8):
            slots, recovered = round_trip(values, bits)
            if recovered[1] != 0:  # 0 is the middle value in each list
                return False
    return True


def test_slots_stay_in_range():
    # The nudge pushed 13 to slot 7.5 -> 8. Clamping keeps it at 7 (the top 3-bit slot).
    slots, recovered = round_trip([-1, 0, 13], 3)
    return slots == [0, 1, 7] and recovered == [-2, 0, 12]


def test_slots_in_range_every_bit_width():
    values = [-1, 0, 2.5, 13]
    for bits in (2, 3, 4, 8):
        slots, _ = round_trip(values, bits)
        if min(slots) < 0 or max(slots) > 2 ** bits - 1:
            return False
    return True


def test_all_positive_values():
    # Values 8..22: 0 is far below. Zero point would be -4, which is fine on paper.
    # We just check every value still comes back close (within one step).
    values = [8, 10, 15, 22]
    slots, recovered = round_trip(values, 3)
    return all(abs(v - r) <= 2 for v, r in zip(values, recovered))


if __name__ == "__main__":
    tests = [test_round_half_up, test_zero_point_whole_steps,
             test_zero_point_needs_a_nudge, test_zero_point_when_start_is_zero,
             test_zero_comes_back_exactly, test_many_zeros_all_exact,
             test_zero_exact_for_odd_ranges, test_slots_stay_in_range,
             test_slots_in_range_every_bit_width, test_all_positive_values]
    passed = 0
    for t in tests:
        ok = t()
        passed += ok
        print(("PASS  " if ok else "FAIL  ") + t.__name__)
    print(f"\n{passed} of {len(tests)} tests passed")
