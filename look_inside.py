# Day 10: Look inside a real model.
#
# We download a tiny sentiment model from Hugging Face,
# pick one grid of weights, and plot them to see their shape.

import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification

# Our own Day 6 code: scale (step), zero point, and the round trip.
from zero_point import (
    get_start_and_step_with_zero,
    get_zero_point,
    quantize,
    dequantize,
)

MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"


def squeeze(weights, bits):
    """Squeeze weights to a few bits and back, using our Day 6 recipe."""
    start, step = get_start_and_step_with_zero(weights, bits)
    zero_point = get_zero_point(start, step)
    buttons = quantize(weights, zero_point, step, bits)
    return dequantize(buttons, zero_point, step)


def pct_vanished(weights, bits):
    """What % of non-zero weights come back as exactly 0?"""
    squeezed = squeeze(weights, bits)
    nonzero = [(w, s) for w, s in zip(weights, squeezed) if w != 0]
    vanished = sum(1 for w, s in nonzero if s == 0)
    return 100 * vanished / len(nonzero)


def describe_weights(weights):
    """Return a few simple facts about a list of weights."""
    count = len(weights)
    smallest = min(weights)
    largest = max(weights)

    # How many weights sit close to 0?
    within_0_1 = sum(1 for w in weights if abs(w) <= 0.1)
    within_1 = sum(1 for w in weights if abs(w) <= 1)
    positive = sum(1 for w in weights if w > 0)

    return {
        "count": count,
        "smallest": smallest,
        "largest": largest,
        "pct_within_0_1": 100 * within_0_1 / count,
        "pct_within_1": 100 * within_1 / count,
        "pct_positive": 100 * positive / count,
    }


def main():
    # Step 1: download the model (about 268 MB, only the first time).
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    # Step 2: grab one grid of weights from the first layer.
    grid = model.distilbert.transformer.layer[0].attention.q_lin.weight
    print("Grid shape:", tuple(grid.shape))

    # Turn the grid into one long, plain list of numbers.
    weights = grid.detach().flatten().tolist()

    # Step 3: print simple facts, then draw the hill.
    facts = describe_weights(weights)
    print(f"Weights in this grid: {facts['count']:,}")
    print(f"Smallest weight:      {facts['smallest']:.3f}")
    print(f"Largest weight:       {facts['largest']:.3f}")
    print(f"Within +/-0.1:        {facts['pct_within_0_1']:.1f}%")
    print(f"Within +/-1:          {facts['pct_within_1']:.1f}%")
    print(f"Positive:             {facts['pct_positive']:.1f}%")

    plt.hist(weights, bins=200, color="steelblue")
    plt.title("590,000 weights from one real grid (layer 1)")
    plt.xlabel("weight value")
    plt.ylabel("how many weights")
    plt.savefig("weights_hill.png", dpi=120)
    print("Saved plot to weights_hill.png")

    # Step 4: squeeze the grid with our own recipe. How many weights vanish?
    print()
    squeezed = squeeze(weights, 4)
    print("First 8 weights at 4 bits:")
    for i in range(8):
        print(f"  {weights[i]:8.4f}  ->  {squeezed[i]:.2f}")
    for bits in [8, 4, 2]:
        print(f"{bits} bits: {pct_vanished(weights, bits):.1f}% of weights vanish to 0")


if __name__ == "__main__":
    main()
