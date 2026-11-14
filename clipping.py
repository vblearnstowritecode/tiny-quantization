# Day 21: Clipping. Cut off the loud judge so the quiet ones get closer stops.
#
# A spike group from Day 20 runs from -6.48 to +0.95. One judge (channel 308)
# sits at -6.48. The other 15 sit near 0. With 16 stops spread over that long
# road, the stops sit far apart, so many quiet judges snap to 0.
#
# Clipping puts up a fence (the cut-off). Anything below the fence moves up to it.
# Then we squeeze as usual. The stops now cover a shorter road, so they sit closer.
#
#   no clip:      -6.48 ●     ●     ●     ●     ●     ●     ●  ... ● +0.95
#   cut at -2:    -2 ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● +0.95
#
# The trade: the loud judge gets one big error (-6.48 reads as -2),
# and the 15 quiet judges each get a smaller error.
# Too loose: quiet judges blurred. Too tight: the loud judge is erased.
# Engineers try many cut-offs and keep the one with the lowest error ("calibration").

from look_inside import squeeze, MODEL_NAME
from fp4 import squeeze_fp4
from activations import get_activations, find_spike_channel, REVIEW, STOP
from per_group import GROUP_SIZE
from race import mean_absolute_error

CUT_OFFS = [-5, -4, -3, -2, -1.5, -1, -0.5]          # for the activation spike group
WEIGHT_CUT_OFFS = [0.4, 0.3, 0.2, 0.15, 0.1, 0.05]    # for the weight grid (both sides)


def clip(group, cut_off):
    # The fence: anything below the cut-off moves up to it. Everyone else stays.
    # Toy example: [-6, 0.2, -0.1, 1] with cut-off -2  ->  [-2, 0.2, -0.1, 1]
    return [max(v, cut_off) for v in group]


def clip_both_sides(values, cut_off):
    # Two fences: one at -cut_off, one at +cut_off. Used for weights, which stick out both ways.
    # Toy example: [-0.7, 0.05, 0.5] with cut-off 0.2  ->  [-0.2, 0.05, 0.2]
    return [min(max(v, -cut_off), cut_off) for v in values]


def clip_then_squeeze(group, cut_off, bits=4):
    # Clip first, then squeeze with our Day 6 linear recipe (min to max, zero point).
    return squeeze(clip(group, cut_off), bits)


def clip_then_squeeze_fp4(group, cut_off):
    # Clip first, then squeeze with FP4 (mirror stops, scale from the biggest size).
    return squeeze_fp4(clip(group, cut_off))


def snapped_to_zero(original, squeezed):
    # How many non-zero numbers came back as exactly 0?
    # Toy example: [0.2, -0.1, 0, 3] -> [0, -0.1, 0, 3]  ->  1 (only the 0.2; the real 0 doesn't count)
    return sum(1 for o, s in zip(original, squeezed) if o != 0 and s == 0)


def best_cut_off(group, cut_offs, bits=4):
    # Calibration: try every cut-off, keep the one with the lowest error.
    # "None" means no clipping at all, so clipping must beat that to win.
    best, best_error = None, mean_absolute_error(group, squeeze(group, bits))
    for cut_off in cut_offs:
        error = mean_absolute_error(group, clip_then_squeeze(group, cut_off, bits))
        if error < best_error:
            best, best_error = cut_off, error
    return best, best_error


def main():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    rows, tokens = get_activations(model, tokenizer, REVIEW)
    spike = find_spike_channel(rows)
    start = (spike // GROUP_SIZE) * GROUP_SIZE
    position = spike - start                   # where the loud judge sits inside the group

    # The same spike group as Day 20.
    token = 3
    group = rows[token][start:start + GROUP_SIZE]
    print(f'\nSpike group: token "{tokens[token]}", channels {start}..{start + GROUP_SIZE - 1}')
    print(f"range {min(group):.2f} .. {max(group):.2f}, loud judge (ch {spike}) = {group[position]:.2f}")

    print("\n4 bits. Error (MAE) over all 16 judges, lower is better.")
    print("cut-off    linear     FP4    quiet judges snapped to 0   loud judge reads")
    lin = squeeze(group, 4)
    fp4 = squeeze_fp4(group)
    print(f"  none    {mean_absolute_error(group, lin):7.3f} {mean_absolute_error(group, fp4):7.3f}"
          f"    {snapped_to_zero(group, lin):2d} of 15 (linear)             {lin[position]:6.2f}")
    for cut_off in CUT_OFFS:
        lin = clip_then_squeeze(group, cut_off)
        fp4 = clip_then_squeeze_fp4(group, cut_off)
        print(f"  {cut_off:4}    {mean_absolute_error(group, lin):7.3f} {mean_absolute_error(group, fp4):7.3f}"
              f"    {snapped_to_zero(group, lin):2d} of 15 (linear)             {lin[position]:6.2f}")

    best, error = best_cut_off(group, CUT_OFFS)
    print(f"\nBest cut-off for linear: {best} (error {error:.3f})")

    # Does the same cut-off win for every token's spike group?
    print("\nBest cut-off for each token's spike group (linear):")
    for t, row in zip(tokens, rows):
        g = row[start:start + GROUP_SIZE]
        b, _ = best_cut_off(g, CUT_OFFS)
        print(f"  {t:>8}  loud judge {g[position]:6.2f}  ->  best cut-off {b}")

    # Second experiment: a whole weight grid with ONE scale (Day 10).
    # Here the loud weights are rare, and the crowd near 0 gets only ~3 stops.
    weights = model.distilbert.transformer.layer[0].attention.q_lin.weight.flatten().tolist()
    print(f"\nWeight grid layer[0] q_lin, one scale for all {len(weights):,} weights")
    print(f"range {min(weights):.3f} .. {max(weights):.3f}")
    print("\n4 bits. Error (MAE), lower is better.")
    print("cut-off      linear    weights snapped to 0   weights pushed to the fence")
    squeezed = squeeze(weights, 4)
    print(f"  none      {mean_absolute_error(weights, squeezed):8.4f}"
          f"     {100 * snapped_to_zero(weights, squeezed) / len(weights):5.1f}%")
    for cut_off in WEIGHT_CUT_OFFS:
        clipped = clip_both_sides(weights, cut_off)
        squeezed = squeeze(clipped, 4)
        pushed = sum(1 for w in weights if abs(w) > cut_off)
        print(f"  +/-{cut_off:<5}  {mean_absolute_error(weights, squeezed):8.4f}"
              f"     {100 * snapped_to_zero(weights, squeezed) / len(weights):5.1f}%"
              f"                {pushed:>7,} ({100 * pushed / len(weights):.2f}%)")


# Run this file to see it:  python clipping.py
if __name__ == "__main__":
    main()
