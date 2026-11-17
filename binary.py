# Day 23: Binary weights. Keep only the sign.
#
# Every weight becomes +1 or -1. That is 1 bit per weight (32x smaller than
# a normal 32-bit number), and multiplying turns into adding or subtracting:
#
#   weight x input  ->   +1 x input = +input
#                        -1 x input = -input      (no multiply needed)
#
# Problem: a tiny 0.21 and a big 0.73 both become 1. Every weight gets
# "loud", like a choir where everyone must sing at full volume.
#
# Fix: one shared number for the whole grid, the average size of the weights
# (signs ignored). Each weight becomes  sign x average size.
# The XNOR-Net paper (2016) uses this same scale.
#
#   weights    [0.21, -0.48, 0.73]
#   signs      [+1,   -1,    +1  ]
#   avg size   (0.21 + 0.48 + 0.73) / 3 = 0.47
#   binary     [+0.47, -0.47, +0.47]

from race import mean_absolute_error
from look_inside import squeeze, MODEL_NAME


def sign(w):
    # Toy example: 0.73 -> 1,  -0.48 -> -1,  0 -> 1
    # 0 has no sign. We send it to +1, so only two values exist (1 bit).
    return 1 if w >= 0 else -1


def binary_plain(weights):
    # Just the signs, no scale.
    # Toy example: [0.21, -0.48, 0.73] -> [1, -1, 1]
    return [sign(w) for w in weights]


def average_size(weights):
    # The average of the sizes, ignoring signs.
    # Toy example: [0.21, -0.48, 0.73] -> (0.21 + 0.48 + 0.73) / 3 = 0.47
    if len(weights) == 0:
        return 0  # no weights: nothing to average
    return sum(abs(w) for w in weights) / len(weights)


def binary_scaled(weights):
    # Each weight becomes  sign x average size.
    # Toy example: [0.21, -0.48, 0.73] -> [0.47, -0.47, 0.47]
    scale = average_size(weights)
    return [sign(w) * scale for w in weights]


def neuron_answer(weights, inputs):
    # Day 9's neuron: multiply each input by its weight, add them up.
    # Toy example: weights [0.21, -0.48, 0.73], inputs [1, 2, 3] -> 1.44
    return sum(w * x for w, x in zip(weights, inputs))


def main():
    # Part 1: Day 9's tiny neuron, by hand.
    inputs = [1, 2, 3]
    weights = [0.21, -0.48, 0.73]
    real = neuron_answer(weights, inputs)
    plain = neuron_answer(binary_plain(weights), inputs)
    scaled = neuron_answer(binary_scaled(weights), inputs)
    print("Part 1: Day 9's neuron, inputs [1, 2, 3], weights [0.21, -0.48, 0.73]")
    print(f"  real answer         {real:.2f}")
    print(f"  plain +-1           {plain:.2f}   miss {abs(plain - real):.2f}")
    print(f"  +-1 x avg size      {scaled:.2f}   miss {abs(scaled - real):.2f}")

    # Part 2: the real q_lin grid from Day 10.
    from transformers import AutoModelForSequenceClassification

    print("\nLoading model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    grid = model.distilbert.transformer.layer[0].attention.q_lin.weight
    weights = grid.detach().flatten().tolist()
    print(f"Part 2: Day 10's grid, {len(weights):,} weights (layer 1, q_lin)\n")

    print(f"  average size of a weight: {average_size(weights):.4f}\n")
    print("                        bits/weight   error (MAE)")
    print(f"  plain +-1                  1         {mean_absolute_error(weights, binary_plain(weights)):.4f}")
    print(f"  +-1 x avg size             1         {mean_absolute_error(weights, binary_scaled(weights)):.4f}")
    print(f"  2-bit linear (Day 11)      2         {mean_absolute_error(weights, squeeze(weights, 2)):.4f}")


# Run this file to see it:  python binary.py
if __name__ == "__main__":
    main()
