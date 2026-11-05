# Day 12: Squeeze the whole model. Do its 👍/👎 answers change?
#
# Step 1 (this part): K-means with a poll.
#   A grid can hold 2.4 million weights. K-means on all of them is slow.
#   So we place the carts using a random sample, like a pollster,
#   and then snap every weight to its nearest cart.

import copy

import numpy as np
import torch
from sklearn.cluster import KMeans

from race import linear_buttons
from look_inside import squeeze


def kmeans_squeeze_sampled(weights, bits, sample_size=50_000, seed=0):
    # 1. Poll: pick a random sample of weights.
    # 2. Place 2^bits carts with K-means, using only the sample.
    # 3. Snap ALL weights to their nearest cart.
    # Toy example: 1 million weights, sample 50,000  ->  carts learned 20x faster
    weights = np.asarray(weights, dtype=np.float64)
    buttons = linear_buttons(weights.tolist(), bits)

    # Fewer different weights than carts? Every weight gets its own cart: no error.
    if len(np.unique(weights)) <= len(buttons):
        return weights.copy()

    # 1. The poll. A fixed seed means the same sample every run.
    rng = np.random.default_rng(seed)
    if len(weights) > sample_size:
        sample = rng.choice(weights, size=sample_size, replace=False)
    else:
        sample = weights

    # 2. Place the carts. They start on the linear buttons, like Day 11.
    start = np.array(buttons).reshape(-1, 1)
    km = KMeans(n_clusters=len(buttons), init=start, n_init=1)
    km.fit(sample.reshape(-1, 1))

    # 3. Everyone walks to their nearest cart.
    carts = km.cluster_centers_.flatten()
    labels = km.predict(weights.reshape(-1, 1))
    return carts[labels]


# Step 2: squeeze every grid in a COPY of the model.
#   The original stays untouched. It is our "before" photo.
#   Each grid gets its own buttons, fitted to its own crowd (per-tensor).

def squeeze_grid(weights, method, bits):
    # Squeeze one grid's weights (a plain list) with "linear" or "kmeans".
    if method == "linear":
        return squeeze(weights, bits)          # our Day 6 recipe
    if method == "kmeans":
        return kmeans_squeeze_sampled(weights, bits).tolist()
    raise ValueError(f"unknown method: {method}")


def squeeze_model(model, method, bits):
    # Gives back a squeezed copy, plus how many grids we squeezed.
    # Only the grids (nn.Linear weights) change. Everything else stays as it was.
    squeezed = copy.deepcopy(model)
    grids = 0
    for layer in squeezed.modules():
        if isinstance(layer, torch.nn.Linear):
            grid = layer.weight.data
            flat = grid.flatten().tolist()
            new_flat = squeeze_grid(flat, method, bits)
            layer.weight.data = torch.tensor(new_flat, dtype=grid.dtype).reshape(grid.shape)
            grids += 1
    return squeezed, grids


# Step 3: ask 20 movie reviews. Count how many answers flip.
#   We compare with the ORIGINAL model's answers, not with a human's.
#   That way, every flip is caused by the squeeze and nothing else.

REVIEWS = [
    # Easy ones
    "I loved every minute of this movie.",
    "A boring, painful two hours.",
    "The acting was brilliant and the story moved me.",
    "Terrible script, terrible acting, terrible everything.",
    "One of the best films I have seen this year.",
    "I walked out halfway through.",
    "A warm, funny and beautiful film.",
    "The plot made no sense at all.",
    # Tricky ones: "not", mixed feelings, sarcasm
    "Not bad at all.",
    "Not good at all.",
    "I expected to hate it, but I loved it.",
    "I expected to love it, but I hated it.",
    "Great actors, wasted on a dull story.",
    "Slow at the start, but the ending is worth it.",
    "Oh great, another two hours of my life gone.",
    "It was fine, I guess.",
    "The music was the only good part.",
    "I would not watch it again, but I don't regret it.",
    "Somehow both too long and too short.",
    "My kids loved it, I fell asleep.",
]


def answers(model, tokenizer, reviews):
    # The model's answer for each review: "pos" (👍) or "neg" (👎).
    # Plain words, because the Windows terminal can't print emoji.
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(reviews, padding=True, return_tensors="pt")
        scores = model(**inputs).logits          # two numbers per review: [👎, 👍]
    return ["pos" if s[1] > s[0] else "neg" for s in scores]


def count_flips(before, after):
    # How many answers changed?
    # Toy example: ["pos", "neg", "pos"] vs ["pos", "pos", "pos"]  ->  1 flip
    return sum(1 for b, a in zip(before, after) if b != a)


def main():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from look_inside import MODEL_NAME

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    original = answers(model, tokenizer, REVIEWS)
    results = {}
    for method in ["linear", "kmeans"]:
        for bits in [8, 4, 2]:
            print(f"Squeezing all grids: {method}, {bits} bits...")
            squeezed, _ = squeeze_model(model, method, bits)
            results[(method, bits)] = answers(squeezed, tokenizer, REVIEWS)

    print(f"\nFlips out of {len(REVIEWS)} reviews (vs the original model)\n")
    print("           8 bits   4 bits   2 bits")
    for method in ["linear", "kmeans"]:
        flips = [count_flips(original, results[(method, b)]) for b in [8, 4, 2]]
        print(f"{method:8s}   {flips[0]:6d}   {flips[1]:6d}   {flips[2]:6d}")

    print("\nreview" + " " * 47 + "orig  lin8 lin4 lin2  km8  km4  km2")
    for i, review in enumerate(REVIEWS):
        row = [results[(m, b)][i] for m in ["linear", "kmeans"] for b in [8, 4, 2]]
        print(f"{review[:50]:50s}   {original[i]}   " + "   ".join(row))


# Run this file to see the whole race:  python whole_model.py
if __name__ == "__main__":
    main()
