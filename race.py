# Day 11: Linear vs K-means on one real layer.
#
# Same 590,000 weights. Same number of buttons. Two ways to place them:
#   Linear:  buttons evenly spaced from smallest to largest weight
#   K-means: buttons (carts) walk to where the weights actually are
# Score: mean absolute error (MAE). Lower wins.

import numpy as np
from sklearn.cluster import KMeans

# Our own Day 6 code for linear, and Day 10's squeeze.
from zero_point import get_start_and_step_with_zero, get_zero_point, dequantize
from look_inside import squeeze, MODEL_NAME


def mean_absolute_error(original, squeezed):
    # Average size of the miss. abs() stops +0.1 and -0.1 from cancelling.
    # Toy example: [0.3, 0.5] vs [0.4, 0.4]  ->  misses 0.1 and 0.1  ->  0.10 (not 0)
    misses = [abs(o - s) for o, s in zip(original, squeezed)]
    return sum(misses) / len(misses)


def linear_buttons(weights, bits):
    # Every button linear can use, as real values.
    # Toy example: weights -2..4, 2 bits  ->  step 2, zero point 1  ->  [-2, 0, 2, 4]
    start, step = get_start_and_step_with_zero(weights, bits)
    zero_point = get_zero_point(start, step)
    all_slots = range(2 ** bits)
    return dequantize(all_slots, zero_point, step)


def kmeans_squeeze(weights, bits):
    # Place 2^bits carts with K-means, then snap every weight to its cart.
    # The carts start on the linear buttons, so K-means begins where linear ends.
    buttons = linear_buttons(weights, bits)

    # Fewer weights than carts? Every weight gets its own cart: no error at all.
    if len(set(weights)) <= len(buttons):
        return list(weights)

    # scikit-learn wants a column: one weight per row.
    column = np.array(weights, dtype=np.float64).reshape(-1, 1)
    start = np.array(buttons, dtype=np.float64).reshape(-1, 1)

    km = KMeans(n_clusters=len(buttons), init=start, n_init=1).fit(column)
    carts = km.cluster_centers_.flatten()
    return [float(carts[label]) for label in km.labels_]


def race(weights, bits_list=(8, 4, 2)):
    # Run both methods at each bit width. Gives back {bits: (linear MAE, kmeans MAE)}.
    scores = {}
    for bits in bits_list:
        linear_mae = mean_absolute_error(weights, squeeze(weights, bits))
        kmeans_mae = mean_absolute_error(weights, kmeans_squeeze(weights, bits))
        scores[bits] = (linear_mae, kmeans_mae)
    return scores


def main():
    from transformers import AutoModelForSequenceClassification

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    grid = model.distilbert.transformer.layer[0].attention.q_lin.weight
    weights = grid.detach().flatten().tolist()
    print(f"Racing on {len(weights):,} weights (layer 1, q_lin)\n")

    print("bits   linear MAE   k-means MAE   k-means error is")
    for bits, (lin, km) in race(weights).items():
        print(f"{bits:4d}   {lin:10.5f}   {km:11.5f}   {100 * km / lin:5.1f}% of linear")


# Run this file to see the race:  python race.py
if __name__ == "__main__":
    main()
