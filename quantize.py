# tiny-quantization: squeeze real numbers into a few whole-number slots.
#
# The whole idea in one line:
#   0.23  ->  divide by scale  ->  round  ->  slot 2  ->  times scale  ->  0.2


def get_scale(weights, top_slot):
    # How much real value is one slot step worth?
    # Toy example: weights [0.2, -0.5, 0.4], top_slot 5  ->  0.5 / 5 = 0.1
    # The biggest weight ignores the minus sign. abs(-0.5) gives 0.5.
    biggest = max(abs(w) for w in weights)
    return biggest / top_slot


def quantize(weights, scale):
    # Turn each real weight into a whole-number slot.
    # Toy example: 0.23 with scale 0.1  ->  round(2.3)  ->  2
    return [round(w / scale) for w in weights]


def dequantize(slots, scale):
    # Turn each slot back into a real number.
    # Toy example: slot 2 with scale 0.1  ->  0.2
    return [s * scale for s in slots]


def top_slot_for(bits):
    # How high can the slots go with this many bits?
    # 8 bits -> 127, 4 bits -> 7, 2 bits -> 1
    # We skip the lone bottom slot (like -128) so both sides mirror each other.
    return 2 ** (bits - 1) - 1


# Run this file to see a round trip:  python quantize.py
if __name__ == "__main__":
    weights = [0.31, -0.82, 0.07, -0.45, 0.66]
    top_slot = 127  # int8 uses slots from -127 to 127

    scale = get_scale(weights, top_slot)
    slots = quantize(weights, scale)
    recovered = dequantize(slots, scale)

    print(f"scale = {scale:.5f}\n")
    print("original   slot   recovered   error")
    for w, s, r in zip(weights, slots, recovered):
        print(f"{w:8.4f}   {s:4d}   {r:9.4f}   {abs(w - r):.4f}")
