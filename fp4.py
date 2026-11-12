# Day 18: FP4's uneven buttons.
#
# Linear puts its 16 buttons at equal gaps.
# FP4 puts them close near 0 and wide far out, like bus stops:
# many stops downtown (where the crowd lives), few in the suburbs.
#
#   value:  0   0.5   1   1.5   2     3     4           6
#           |----|----|----|----|-----|-----|-----------|
#   gap:      0.5  0.5  0.5  0.5   1     1        2
#
# The buttons are fixed and built into the chip, so no lookup table is stored.
# A scale stretches them to fit the weights:
#   weight -> divide by scale -> snap to the nearest button -> multiply by scale

from race import mean_absolute_error
from look_inside import squeeze, MODEL_NAME
from per_group import cut_into_groups, squeeze_per_group, GROUP_SIZE

# The 8 positive FP4 values, and their mirror on the negative side.
FP4_POSITIVE = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
FP4_BUTTONS = sorted(set([-b for b in FP4_POSITIVE] + FP4_POSITIVE))  # 15 buttons

TOP_BUTTON = 6


def nearest_button(x):
    # Toy example: 2.004 -> 2,   4.5 -> 4 (4 is 0.5 away, 6 is 1.5 away)
    # On a tie (like 2.5, halfway between 2 and 3) we keep the first one we meet.
    best = FP4_BUTTONS[0]
    for b in FP4_BUTTONS:
        if abs(x - b) < abs(x - best):
            best = b
    return best


def fp4_scale(weights):
    # Pick the scale so the biggest weight (ignoring sign) lands on button 6.
    # Toy example: weights 100, 200.4, 450, 600  ->  600 / 6 = 100
    biggest = max(abs(w) for w in weights)
    if biggest == 0:
        return 1  # all zeros: any scale works, and 0 stays 0
    return biggest / TOP_BUTTON


def squeeze_fp4(weights):
    # The full round trip: divide, snap, multiply.
    # Toy example (scale 100): 100 -> 100,  200.4 -> 200,  450 -> 400,  600 -> 600
    scale = fp4_scale(weights)
    return [nearest_button(w / scale) * scale for w in weights]


def squeeze_fp4_per_group(rows, group_size=GROUP_SIZE):
    # Day 19: same as Day 15's per-group, but each group uses FP4 buttons.
    # Cut each row into groups of 16. Each group gets its own FP4 scale.
    # Toy example, group size 2:  [[1, 6, 0.5, 3]]  ->  groups [1, 6] and [0.5, 3]
    #   group [1, 6]:    scale 1,   so 1 -> 1,     6 -> 6
    #   group [0.5, 3]:  scale 0.5, so 0.5 -> 0.5, 3 -> 3
    new_rows = []
    for row in rows:
        new_row = []
        for group in cut_into_groups(row, group_size):
            new_row.extend(squeeze_fp4(group))
        new_rows.append(new_row)
    return new_rows


def crowd_buttons(squeezed, crowd=0.1):
    # How many different buttons do the weights near 0 (within +/-0.1) land on?
    # More buttons in the crowd = finer detail where most weights live.
    return len(set(s for s in squeezed if abs(s) <= crowd))


def main():
    from transformers import AutoModelForSequenceClassification

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    grid = model.distilbert.transformer.layer[0].attention.q_lin.weight
    weights = grid.detach().flatten().tolist()
    print(f"Same grid as Day 10: {len(weights):,} weights (layer 1, q_lin)\n")

    linear = squeeze(weights, 4)
    fp4 = squeeze_fp4(weights)

    print("4 bits, one scale for the whole grid")
    print("                 linear (even gaps)   FP4 (close near 0)")
    print(f"error (MAE)      {mean_absolute_error(weights, linear):10.5f}"
          f"           {mean_absolute_error(weights, fp4):10.5f}")
    print(f"crowd buttons    {crowd_buttons(linear):10d}"
          f"           {crowd_buttons(fp4):10d}")
    print(f"\nFP4 scale = {fp4_scale(weights):.4f}, so its buttons near 0 sit at:")
    print("  " + ", ".join(f"{b * fp4_scale(weights):.3f}" for b in FP4_BUTTONS if b >= 0))

    # Day 19: give both of them groups of 16 (like Day 15). Does FP4 still win?
    rows = grid.detach().tolist()
    linear_groups = [w for row in squeeze_per_group(rows, 4) for w in row]
    fp4_groups = [w for row in squeeze_fp4_per_group(rows) for w in row]

    print(f"\n4 bits, groups of {GROUP_SIZE} (each group gets its own scale)")
    print("                 linear (even gaps)   FP4 (close near 0)")
    print(f"error (MAE)      {mean_absolute_error(weights, linear_groups):10.5f}"
          f"           {mean_absolute_error(weights, fp4_groups):10.5f}")


# Run this file to see the race:  python fp4.py
if __name__ == "__main__":
    main()
