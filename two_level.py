# Day 16: two-level scaling. Shrink the scale bill by quantizing the scales too.
#
# The whole idea in one line:
#   Day 15: every group of 16 stores a 32-bit scale  ->  32 / 16 = 2 extra bits per weight
#   Day 16: every group stores an 8-bit scale button ->   8 / 16 = 0.5 extra bits per weight
#           plus ONE 32-bit "scale for the scales" per grid (shared by ~590k weights, so ~0)
#
# Packing goes up, then down:
#   1. find each group's scale               (same as Day 15)
#   2. find the grid scale from those scales (biggest scale / 255)
#   3. turn each group scale into a button   (round(scale / grid scale))
#   4. quantize the weights with the SQUEEZED scales, because that's all we keep
#
# Unpacking goes down only:
#   grid scale x scale button = group scale,   group scale x weight button = weight

import torch

from per_group import cut_into_groups, GROUP_SIZE
from zero_point import (round_half_up, get_start_and_step_with_zero,
                        get_zero_point, quantize, dequantize)

SCALE_BITS = 8


def squeeze_scales(scales, top_button):
    # Quantize the scales themselves. Scales are never negative, so 0 is the floor.
    # Toy example (from our hand-worked one), buttons 0..10:
    #   scales [0.30, 0.04]  ->  grid scale 0.30 / 10 = 0.03
    #   buttons [10, 1]      ->  back [0.30, 0.03]   (B lost 0.01)
    grid_scale = max(scales) / top_button
    buttons = []
    for s in scales:
        b = round_half_up(s / grid_scale)
        # A scale of 0 would make every weight in the group 0. Keep at least button 1.
        b = max(1, min(top_button, b))
        buttons.append(b)
    return buttons, grid_scale


def squeeze_grid_two_level(rows, bits, group_size=GROUP_SIZE, scale_bits=SCALE_BITS):
    # Step 1: cut every row into groups and find each group's start and scale.
    groups = []
    starts = []
    scales = []
    for row in rows:
        for group in cut_into_groups(row, group_size):
            start, step = get_start_and_step_with_zero(group, bits)
            groups.append(group)
            starts.append(start)
            scales.append(step)

    # Steps 2 and 3: one grid scale, and an 8-bit button for every group scale.
    buttons, grid_scale = squeeze_scales(scales, 2 ** scale_bits - 1)

    # Step 4: squeeze each group's weights with its SQUEEZED scale.
    flat = []
    for group, start, button in zip(groups, starts, buttons):
        step = button * grid_scale
        zero_point = get_zero_point(start, step)
        slots = quantize(group, zero_point, step, bits)
        flat.extend(dequantize(slots, zero_point, step))

    # Glue the weights back into rows of the original length.
    new_rows = []
    i = 0
    for row in rows:
        new_rows.append(flat[i:i + len(row)])
        i += len(row)
    return new_rows


def scale_bits_per_weight(group_size=GROUP_SIZE, scale_bits=SCALE_BITS):
    # Each weight's share of its group's scale button (the dinner bill again).
    # The one grid scale is shared by so many weights that we leave it out.
    return scale_bits / group_size


def squeeze_model_two_level(model, bits, group_size=GROUP_SIZE, scale_bits=SCALE_BITS):
    # Same as Day 15's squeeze_model_per_group, but with two-level scales.
    # Each grid (nn.Linear) gets its own grid scale.
    import copy
    squeezed = copy.deepcopy(model)
    grids = 0
    for layer in squeezed.modules():
        if isinstance(layer, torch.nn.Linear):
            grid = layer.weight.data
            new_rows = squeeze_grid_two_level(grid.tolist(), bits, group_size, scale_bits)
            layer.weight.data = torch.tensor(new_rows, dtype=grid.dtype)
            grids += 1
    return squeezed, grids


def main():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from look_inside import MODEL_NAME
    from race import mean_absolute_error
    from per_group import squeeze_per_group
    from whole_model import answers, count_flips, REVIEWS

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Part 1: one grid. How much error do 8-bit scales add?
    grid = model.distilbert.transformer.layer[0].attention.q_lin.weight.data
    rows = grid.tolist()
    flat = grid.flatten().tolist()
    print(f"\nOne grid (layer 0, q_lin): average error per weight (groups of {GROUP_SIZE})")
    print("bits   32-bit scales   8-bit scales")
    for bits in [8, 4, 2]:
        day15 = [w for row in squeeze_per_group(rows, bits) for w in row]
        today = [w for row in squeeze_grid_two_level(rows, bits) for w in row]
        print(f"{bits:4d}   {mean_absolute_error(flat, day15):13.5f}   "
              f"{mean_absolute_error(flat, today):12.5f}")

    # Part 2: the whole model. Do the answers still hold?
    original = answers(model, tokenizer, REVIEWS)
    results = {}
    for bits in [8, 4, 2]:
        print(f"Squeezing all grids at {bits} bits, two-level...")
        results[bits] = answers(squeeze_model_two_level(model, bits)[0], tokenizer, REVIEWS)

    print(f"\nFlips out of {len(REVIEWS)} reviews (vs the original model)\n")
    print("                          8 bits   4 bits   2 bits   scale bits per weight")
    print(f"per-group, 32-bit scales       0        0        1   "
          f"{32 / GROUP_SIZE:.2f}   (Day 15)")
    flips = [count_flips(original, results[b]) for b in [8, 4, 2]]
    print(f"two-level, 8-bit scales   {flips[0]:6d}   {flips[1]:6d}   {flips[2]:6d}   "
          f"{scale_bits_per_weight():.2f}   (today)")

    print("\nreview" + " " * 47 + "orig  t8   t4   t2")
    for i, review in enumerate(REVIEWS):
        row = [results[b][i] for b in [8, 4, 2]]
        print(f"{review[:50]:50s}   {original[i]}  " + "  ".join(row))


# Run this file to see two-level results:  python two_level.py
if __name__ == "__main__":
    main()
