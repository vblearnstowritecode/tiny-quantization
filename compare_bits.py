# Day 2: what happens when we use fewer bits?
# Run with:  python compare_bits.py

from quantize import get_scale, quantize, dequantize, top_slot_for

weights = [0.31, -0.82, 0.07, -0.45, 0.66]

# Part 1: the big picture. Fewer bits -> bigger steps -> bigger errors.
print("bits   slots used     step    worst possible error")
for bits in [8, 4, 2]:
    top = top_slot_for(bits)
    scale = get_scale(weights, top)
    print(f"{bits:4d}   {-top:4d} to {top:<4d}  {scale:.3f}   {scale / 2:.3f}")

# Part 2: where each weight ends up, for each bit width.
print("\nweight      8-bit     4-bit     2-bit")
recovered = {}
for bits in [8, 4, 2]:
    scale = get_scale(weights, top_slot_for(bits))
    recovered[bits] = dequantize(quantize(weights, scale), scale)

for i, w in enumerate(weights):
    print(f"{w:6.2f}   {recovered[8][i]:7.3f}   {recovered[4][i]:7.3f}   {recovered[2][i]:7.3f}")
