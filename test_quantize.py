# Checks for quantize.py. Run with:  python test_quantize.py
# Each test prints PASS or FAIL, so you can see exactly what works.

from quantize import get_scale, quantize, dequantize, top_slot_for


def close(a, b):
    # Computers store decimals slightly imperfectly (0.1 + 0.2 = 0.30000000000000004),
    # so we check "very close" instead of "exactly equal".
    return abs(a - b) < 1e-9


def test_scale_toy_example():
    # The lift example: biggest weight 0.5, top slot 5, so each step is 0.1
    return close(get_scale([0.2, -0.5, 0.4, 0.1], 5), 0.1)


def test_scale_ignores_minus_sign():
    # -0.9 is the biggest by size, even though it is negative
    return close(get_scale([0.3, -0.9], 9), 0.1)


def test_quantize_exact_fit():
    # 0.2 lands right on slot 2
    return quantize([0.2], 0.1) == [2]


def test_quantize_rounds_to_nearest():
    # 0.23 -> 2.3 -> slot 2, and 0.27 -> 2.7 -> slot 3
    return quantize([0.23, 0.27], 0.1) == [2, 3]


def test_quantize_negative():
    # Negative weights go to negative slots
    return quantize([-0.5, -0.23], 0.1) == [-5, -2]


def test_dequantize():
    # Slot 2 means 0.2, slot -5 means -0.5
    result = dequantize([2, -5], 0.1)
    return close(result[0], 0.2) and close(result[1], -0.5)


def test_round_trip_loses_a_little():
    # 0.23 goes in, 0.2 comes out. We lose 0.03.
    scale = 0.1
    back = dequantize(quantize([0.23], scale), scale)
    return close(back[0], 0.2)


def test_biggest_weight_hits_top_slot():
    # The biggest weight should land exactly on the top slot
    weights = [0.31, -0.82, 0.07, -0.45, 0.66]
    slots = quantize(weights, get_scale(weights, 127))
    return slots[1] == -127


def test_slots_stay_in_range():
    # No slot may go past -127 or 127, or it won't fit in 8 bits
    weights = [0.31, -0.82, 0.07, -0.45, 0.66, 0.819, -0.001]
    slots = quantize(weights, get_scale(weights, 127))
    return all(-127 <= s <= 127 for s in slots)


def test_error_never_beyond_half_a_step():
    # Your finding: the worst loss is half a step. Try many weights.
    weights = [i / 1000 for i in range(-1000, 1001)]  # -1.0, -0.999, ... 1.0
    scale = get_scale(weights, 127)
    recovered = dequantize(quantize(weights, scale), scale)
    worst = max(abs(w - r) for w, r in zip(weights, recovered))
    return worst <= scale / 2 + 1e-9


# ---- Day 2: fewer bits ----

def test_top_slot_for_each_bit_width():
    # Your formula: 2^(bits-1) - 1
    return top_slot_for(8) == 127 and top_slot_for(4) == 7 and top_slot_for(2) == 1


def test_4_bit_slots_stay_in_range():
    weights = [0.31, -0.82, 0.07, -0.45, 0.66]
    slots = quantize(weights, get_scale(weights, top_slot_for(4)))
    return all(-7 <= s <= 7 for s in slots)


def test_2_bit_uses_only_three_slots():
    # 2 bits leaves only -1, 0 and 1 (ternary)
    weights = [i / 100 for i in range(-100, 101)]
    slots = quantize(weights, get_scale(weights, top_slot_for(2)))
    return set(slots) == {-1, 0, 1}


def test_2_bit_worked_example():
    # From our session: 0.31 -> 0, 0.66 -> 0.82, -0.45 -> -0.82
    weights = [0.31, -0.82, 0.07, -0.45, 0.66]
    scale = get_scale(weights, top_slot_for(2))
    back = dequantize(quantize(weights, scale), scale)
    expected = [0.0, -0.82, 0.0, -0.82, 0.82]
    return all(close(b, e) for b, e in zip(back, expected))


def test_half_a_step_holds_for_every_bit_width():
    # The fence rule works no matter how far apart the posts are
    weights = [i / 1000 for i in range(-1000, 1001)]
    for bits in [8, 4, 2]:
        scale = get_scale(weights, top_slot_for(bits))
        recovered = dequantize(quantize(weights, scale), scale)
        worst = max(abs(w - r) for w, r in zip(weights, recovered))
        if worst > scale / 2 + 1e-9:
            return False
    return True


def test_fewer_bits_means_bigger_error():
    # 8 bits beats 4 bits, and 4 bits beats 2 bits
    weights = [i / 1000 for i in range(-1000, 1001)]
    worst = {}
    for bits in [8, 4, 2]:
        scale = get_scale(weights, top_slot_for(bits))
        recovered = dequantize(quantize(weights, scale), scale)
        worst[bits] = max(abs(w - r) for w, r in zip(weights, recovered))
    return worst[8] < worst[4] < worst[2]


# ---- Day 3: fewer bits can delete weights ----

def count_zeros(weights, bits):
    slots = quantize(weights, get_scale(weights, top_slot_for(bits)))
    return slots.count(0)


def test_031_becomes_zero_at_2_bits():
    # Your answer: 0.31 / 0.82 = 0.38, less than half a step, so slot 0
    weights = [0.31, -0.82, 0.07, -0.45, 0.66]
    slots = quantize(weights, get_scale(weights, top_slot_for(2)))
    return slots[0] == 0


def test_minus_045_survives_at_2_bits():
    # -0.45 / 0.82 = -0.55, just past half a step, so slot -1 (not zero)
    weights = [0.31, -0.82, 0.07, -0.45, 0.66]
    slots = quantize(weights, get_scale(weights, top_slot_for(2)))
    return slots[3] == -1


def test_zero_count_for_our_weights():
    # 8 and 4 bits keep all five. 2 bits deletes 0.07 and 0.31.
    weights = [0.31, -0.82, 0.07, -0.45, 0.66]
    return (count_zeros(weights, 8) == 0
            and count_zeros(weights, 4) == 0
            and count_zeros(weights, 2) == 2)


def test_fewer_bits_never_means_fewer_zeros():
    # Taller steps make a wider zero zone, so zeros can only grow
    weights = [i / 1000 for i in range(-1000, 1001)]
    return count_zeros(weights, 8) <= count_zeros(weights, 4) <= count_zeros(weights, 2)


def test_zero_exactly_when_inside_half_a_step():
    # The rule: a weight becomes zero when it is closer than half a step to 0
    weights = [i / 1000 for i in range(-1000, 1001)]
    for bits in [8, 4, 2]:
        scale = get_scale(weights, top_slot_for(bits))
        slots = quantize(weights, scale)
        for w, s in zip(weights, slots):
            if abs(abs(w) - scale / 2) < 1e-9:
                continue  # exactly on the edge: a tie, skip it
            if (s == 0) != (abs(w) < scale / 2):
                return False
    return True


if __name__ == "__main__":
    tests = [
        test_scale_toy_example,
        test_scale_ignores_minus_sign,
        test_quantize_exact_fit,
        test_quantize_rounds_to_nearest,
        test_quantize_negative,
        test_dequantize,
        test_round_trip_loses_a_little,
        test_biggest_weight_hits_top_slot,
        test_slots_stay_in_range,
        test_error_never_beyond_half_a_step,
        test_top_slot_for_each_bit_width,
        test_4_bit_slots_stay_in_range,
        test_2_bit_uses_only_three_slots,
        test_2_bit_worked_example,
        test_half_a_step_holds_for_every_bit_width,
        test_fewer_bits_means_bigger_error,
        test_031_becomes_zero_at_2_bits,
        test_minus_045_survives_at_2_bits,
        test_zero_count_for_our_weights,
        test_fewer_bits_never_means_fewer_zeros,
        test_zero_exactly_when_inside_half_a_step,
    ]
    passed = 0
    for test in tests:
        try:
            ok = test()
        except Exception as e:
            ok = False
            print(f"FAIL  {test.__name__}  (crashed: {e})")
            continue
        print(("PASS  " if ok else "FAIL  ") + test.__name__)
        passed += ok
    print(f"\n{passed} of {len(tests)} tests passed")
