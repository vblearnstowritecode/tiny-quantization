# Day 6: make real 0 land exactly on a slot.
#
# The whole idea in one line:
#   values -1..13, step 2  ->  0 falls between slots  ->  nudge start to -2  ->  0 sits on slot 1
#
# Networks are full of exact zeros (ReLU, pruning, padding).
# If 0 has no slot, every one of those zeros comes back wrong.
# The slot that holds real 0 is called the zero point.

from shift_quantize import get_start_and_step


def round_half_up(x):
    # Python's round(0.5) gives 0, which surprises people. We want 0.5 -> 1.
    return int(x + 0.5) if x >= 0 else -int(-x + 0.5)


def get_zero_point(start, step):
    # How many steps to climb from start up to 0?
    # Toy example: start -6, step 2  ->  6 / 2  ->  zero point 3
    # If it's not a whole number (start -1, step 2 -> 0.5), we round to a nearby slot.
    return round_half_up(-start / step)


def quantize(values, zero_point, step, bits):
    # Same as Day 5, but we stand on the zero point instead of the start.
    # Toy example: 4 with zero point 1, step 2  ->  4 / 2 + 1  ->  slot 3
    top_slot = 2 ** bits - 1
    slots = []
    for v in values:
        s = round_half_up(v / step) + zero_point
        # The nudge can push the biggest value one slot too high. Clamp it back.
        s = max(0, min(top_slot, s))
        slots.append(s)
    return slots


def dequantize(slots, zero_point, step):
    # Turn each slot back into a real value: how far is it from the zero point?
    # Toy example: slot 3 with zero point 1, step 2  ->  2 * (3 - 1)  ->  4
    return [step * (s - zero_point) for s in slots]


# Run this file to see a round trip:  python zero_point.py
if __name__ == "__main__":
    values = [-1, 0, 0, 4, 0, 7, 13]
    bits = 3

    start, step = get_start_and_step(values, bits)
    zero_point = get_zero_point(start, step)
    slots = quantize(values, zero_point, step, bits)
    recovered = dequantize(slots, zero_point, step)

    print(f"start = {start}, step = {step}, zero point = {zero_point}\n")
    print("original   slot   recovered   error")
    for v, s, r in zip(values, slots, recovered):
        print(f"{v:8.3f}   {s:4d}   {r:9.3f}   {abs(v - r):.3f}")
