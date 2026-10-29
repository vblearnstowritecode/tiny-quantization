# Checks for shift_quantize.py. Run with:  python test_shift_quantize.py
# Each test prints PASS or FAIL, so you can see exactly what works.

from shift_quantize import get_start_and_step, quantize, dequantize
import quantize as day1  # the Day 1 method, to compare against


def close(a, b):
    # Computers store decimals slightly imperfectly, so we check "very close".
    return abs(a - b) < 1e-9


def test_start_and_step_toy_example():
    # Our example from the session: 8..22 with 3 bits -> start 8, step 2
    start, step = get_start_and_step([8, 10, 12, 14, 16, 18, 20, 22], 3)
    return close(start, 8) and close(step, 2)


def test_order_does_not_matter():
    # Shuffled values give the same notes
    start, step = get_start_and_step([14, 22, 8, 18], 3)
    return close(start, 8) and close(step, 2)


def test_step_from_8_to_14():
    # The first example: 8..14 with 3 bits -> step 6 / 7 = 0.857
    start, step = get_start_and_step([8, 10, 12, 14], 3)
    return close(start, 8) and close(step, 6 / 7)


def test_negative_start():
    # Values can start below 0: -2..12 with 3 bits -> start -2, step 14 / 7 = 2
    start, step = get_start_and_step([-2, 0, 4, 12], 3)
    return close(start, -2) and close(step, 2)


def test_steps_for_each_bit_width():
    # Your rule: 2 ** bits - 1 steps. Height 0..255 makes the step easy to read.
    values = [0, 255]
    return (close(get_start_and_step(values, 8)[1], 1)       # 255 steps
            and close(get_start_and_step(values, 4)[1], 17)  # 15 steps
            and close(get_start_and_step(values, 2)[1], 85)) # 3 steps


def test_all_same_values_do_not_crash():
    # Your catch: 5, 5, 5 gives a step of 0. We use 1 instead, and all land on slot 0.
    start, step = get_start_and_step([5, 5, 5], 3)
    return step == 1 and quantize([5, 5, 5], start, step) == [0, 0, 0]


# ---- Function 2: value -> slot ----

def test_quantize_toy_example():
    # Your hand answer: (14 - 8) / 2 = 3
    return quantize([14], 8, 2) == [3]


def test_quantize_fills_every_slot():
    # 8..22 with 3 bits uses all 8 slots, 0 to 7. Nothing wasted.
    values = [8, 10, 12, 14, 16, 18, 20, 22]
    start, step = get_start_and_step(values, 3)
    return quantize(values, start, step) == [0, 1, 2, 3, 4, 5, 6, 7]


def test_quantize_rounds_to_nearest():
    # 14.9 -> 3.45 -> slot 3, and 15.1 -> 3.55 -> slot 4
    return quantize([14.9, 15.1], 8, 2) == [3, 4]


def test_quantize_negative_start():
    # -2..12: real 0 sits on slot 1 (the zero point from our session)
    return quantize([-2, 0, 4, 12], -2, 2) == [0, 1, 3, 7]


def test_slots_stay_in_range():
    # Many messy values, 3 bits: every slot must be between 0 and 7
    values = [8 + i / 100 for i in range(0, 1401)]  # 8.00, 8.01, ... 22.00
    start, step = get_start_and_step(values, 3)
    return all(0 <= s <= 7 for s in quantize(values, start, step))


# ---- Function 3: slot -> value ----

def test_dequantize_toy_example():
    # Your hand answers: slot 0 -> 8, slot 5 -> 18, slot 7 -> 22
    return dequantize([0, 5, 7], 8, 2) == [8, 18, 22]


def test_round_trip_exact_on_a_step():
    # Values that sit right on a step come back perfectly
    values = [8, 10, 12, 14, 16, 18, 20, 22]
    start, step = get_start_and_step(values, 3)
    return dequantize(quantize(values, start, step), start, step) == values


def test_round_trip_loses_a_little():
    # 14.9 lands on slot 3 and comes back as 14. We lose 0.9.
    back = dequantize(quantize([14.9], 8, 2), 8, 2)
    return close(back[0], 14)


def test_error_never_beyond_half_a_step():
    # The Day 1 rule still holds: the worst loss is half a step
    values = [8 + i / 100 for i in range(0, 601)]  # 8.00 ... 14.00
    for bits in [8, 4, 3, 2]:
        start, step = get_start_and_step(values, bits)
        back = dequantize(quantize(values, start, step), start, step)
        worst = max(abs(v - b) for v, b in zip(values, back))
        if worst > step / 2 + 1e-9:
            return False
    return True


def test_beats_day_1_on_values_far_from_zero():
    # Today's win. Values 8..14, 3 bits. Day 1 wastes slots below 8; we don't.
    values = [8 + i / 100 for i in range(0, 601)]

    start, step = get_start_and_step(values, 3)
    back = dequantize(quantize(values, start, step), start, step)
    worst_today = max(abs(v - b) for v, b in zip(values, back))

    scale = day1.get_scale(values, day1.top_slot_for(3))
    back1 = day1.dequantize(day1.quantize(values, scale), scale)
    worst_day1 = max(abs(v - b) for v, b in zip(values, back1))

    return worst_today < worst_day1


if __name__ == "__main__":
    tests = [
        test_start_and_step_toy_example,
        test_order_does_not_matter,
        test_step_from_8_to_14,
        test_negative_start,
        test_steps_for_each_bit_width,
        test_all_same_values_do_not_crash,
        test_quantize_toy_example,
        test_quantize_fills_every_slot,
        test_quantize_rounds_to_nearest,
        test_quantize_negative_start,
        test_slots_stay_in_range,
        test_dequantize_toy_example,
        test_round_trip_exact_on_a_step,
        test_round_trip_loses_a_little,
        test_error_never_beyond_half_a_step,
        test_beats_day_1_on_values_far_from_zero,
    ]
    passed = 0
    for test in tests:
        result = test()
        passed += result
        print(f"{'PASS' if result else 'FAIL'}  {test.__name__}")
    print(f"\n{passed} of {len(tests)} passed")
