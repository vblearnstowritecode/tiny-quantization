# Day 5: shift the staircase so it starts at the smallest value.
#
# The whole idea in one line:
#   values 8..22  ->  start at 8, not at 0  ->  all 8 slots get used
#
# Day 1 started the slots at 0, so values like 8..22 left the bottom slots empty.
# Here slot 0 sits on the smallest value, and the top slot sits on the biggest.


def get_start_and_step(values, bits):
    # Find the two notes we need: where the staircase starts, and how tall a step is.
    # Toy example: values 8..22, 3 bits  ->  start 8, step (22 - 8) / 7 = 2
    start = min(values)
    steps = 2 ** bits - 1  # 3 bits -> 8 slots -> 7 steps between them
    step = (max(values) - start) / steps

    # If every value is the same (like 5, 5, 5), the step is 0 and we can't divide by it.
    # Any step works then, because every value sits on slot 0. We pick 1.
    if step == 0:
        step = 1

    return start, step


def quantize(values, start, step):
    # Turn each real value into a whole-number slot.
    # Toy example: 14 with start 8, step 2  ->  round((14 - 8) / 2)  ->  slot 3
    return [round((v - start) / step) for v in values]


def dequantize(slots, start, step):
    # Turn each slot back into a real value: stand on the start, then climb.
    # Toy example: slot 3 with start 8, step 2  ->  8 + 2 * 3  ->  14
    return [start + step * s for s in slots]


# Run this file to see a round trip:  python shift_quantize.py
if __name__ == "__main__":
    values = [8, 9.3, 10.6, 12.2, 14]
    bits = 3

    start, step = get_start_and_step(values, bits)
    slots = quantize(values, start, step)
    recovered = dequantize(slots, start, step)

    print(f"start = {start}, step = {step:.3f}\n")
    print("original   slot   recovered   error")
    for v, s, r in zip(values, slots, recovered):
        print(f"{v:8.3f}   {s:4d}   {r:9.3f}   {abs(v - r):.3f}")
