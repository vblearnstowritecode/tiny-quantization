# Day 3: with few bits, small weights quietly become zero.
# Run with:  python zero_weights.py

from quantize import get_scale, quantize, top_slot_for

weights = [0.31, -0.82, 0.07, -0.45, 0.66]

# A weight becomes zero when it sits less than half a step away from 0.
print("bits    step   zero zone           weights turned to zero")
for bits in [8, 4, 2]:
    scale = get_scale(weights, top_slot_for(bits))
    slots = quantize(weights, scale)
    zeros = [w for w, s in zip(weights, slots) if s == 0]
    zone = f"-{scale / 2:.3f} to {scale / 2:.3f}"
    print(f"{bits:4d}   {scale:.3f}   {zone:<18}  {len(zeros)} of {len(weights)}  {zeros}")
