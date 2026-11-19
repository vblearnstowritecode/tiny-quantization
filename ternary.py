# Day 25: Ternary weights. Binary plus a 0 button.
#
# Binary (Day 23) forces every weight to +s or -s. Quiet weights near 0
# become loud, like a choir where whispers turn into shouts.
#
# Ternary adds a third button at 0, so quiet weights can stay silent:
#
#        -s              0              +s
#         *--------------*--------------*
#
# The rule (from the Ternary Weight Networks paper, 2016):
#   line = 0.7 x average size of all weights
#   size below the line  -> 0
#   size above the line  -> keep the sign, use s
#   s    = average size of the weights above the line
#
# Toy example:
#   weights   [0.9, -0.8, 0.02, -0.01, 0.03, 0.01]
#   avg size  1.77 / 6 = 0.295,  line = 0.7 x 0.295 = 0.2065
#   above     0.9 and 0.8,  s = (0.9 + 0.8) / 2 = 0.85
#   ternary   [0.85, -0.85, 0, 0, 0, 0]

from race import mean_absolute_error
from look_inside import squeeze, MODEL_NAME
from binary import sign, average_size, binary_scaled


def ternary_line(weights):
    # The cut-off: 0.7 x the average size.
    # Toy example: [0.9, -0.8, 0.02, -0.01, 0.03, 0.01] -> 0.7 x 0.295 = 0.2065
    return 0.7 * average_size(weights)


def ternary_scale(weights, line):
    # s = the average size of the weights above the line (the loud ones).
    # Toy example: line 0.2065, loud ones 0.9 and 0.8 -> s = 0.85
    loud = [w for w in weights if abs(w) > line]
    return average_size(loud)  # 0 if nobody is loud


def ternary(weights):
    # Each weight becomes -s, 0 or +s.
    # Toy example: [0.9, -0.8, 0.02, -0.01, 0.03, 0.01]
    #           -> [0.85, -0.85, 0, 0, 0, 0]
    line = ternary_line(weights)
    s = ternary_scale(weights, line)
    out = []
    for w in weights:
        if abs(w) > line:
            out.append(sign(w) * s)  # loud: keep the sign
        else:
            out.append(0)  # quiet: stay silent
    return out


def share_of_zeros(values):
    # What fraction of values are exactly 0.
    # Toy example: [0.85, -0.85, 0, 0, 0, 0] -> 4 / 6 = 0.67
    if len(values) == 0:
        return 0
    return sum(1 for v in values if v == 0) / len(values)


def main():
    # Part 1: the choir toy. 2 loud singers, 4 quiet ones.
    weights = [0.9, -0.8, 0.02, -0.01, 0.03, 0.01]
    b = binary_scaled(weights)
    t = ternary(weights)
    print("Part 1: the choir toy")
    print(f"  weights   {weights}")
    print(f"  binary    {[round(x, 3) for x in b]}   error {mean_absolute_error(weights, b):.4f}")
    print(f"  ternary   {[round(x, 3) for x in t]}   error {mean_absolute_error(weights, t):.4f}")

    # Part 2: the real q_lin grid from Day 10.
    from transformers import AutoModelForSequenceClassification

    print("\nLoading model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    grid = model.distilbert.transformer.layer[0].attention.q_lin.weight
    weights = grid.detach().flatten().tolist()
    print(f"Part 2: Day 10's grid, {len(weights):,} weights (layer 1, q_lin)\n")

    line = ternary_line(weights)
    s = ternary_scale(weights, line)
    t = ternary(weights)
    print(f"  line = {line:.4f},  s = {s:.4f}")
    print(f"  weights sent to 0: {share_of_zeros(t):.1%}\n")

    print("                        buttons   error (MAE)")
    print(f"  binary  +-avg (Day 23)    2      {mean_absolute_error(weights, binary_scaled(weights)):.4f}")
    print(f"  2-bit linear (Day 11)     4      {mean_absolute_error(weights, squeeze(weights, 2)):.4f}")
    print(f"  ternary  -s, 0, +s        3      {mean_absolute_error(weights, t):.4f}")


# Run this file to see it:  python ternary.py
if __name__ == "__main__":
    main()
