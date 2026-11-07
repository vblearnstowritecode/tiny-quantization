# Day 14: per-channel. Give every row of a grid its own buttons.
#
# The whole idea in one line:
#   one outlier in row 3  ->  per-tensor: the whole grid gets wide steps
#                         ->  per-channel: only row 3 gets wide steps
#
# A layer is a panel of judges. Each row of the grid is one judge's
# scoring sheet. Each judge has their own habits, so each row gets its own
# scale and zero point. Cost: one extra scale per row (about 1% more memory).

import torch

from look_inside import squeeze


def squeeze_per_channel(rows, bits):
    # Squeeze each row by itself, with the same Day 6 recipe.
    # Toy example, 2 bits:
    #   rows [[0.1, 0.2, 0.3], [-5, 0, 5]]
    #   per-tensor:  the -5..5 range sets one wide step for BOTH rows -> row 1 becomes all 0
    #   per-channel: row 1 gets its own small step              -> row 1 stays [0.1, 0.2, 0.3]
    return [squeeze(row, bits) for row in rows]


def squeeze_model_per_channel(model, bits):
    # Same as Day 12's squeeze_model, but row by row instead of the whole grid at once.
    # Gives back a squeezed copy, plus how many grids we squeezed.
    import copy
    squeezed = copy.deepcopy(model)
    grids = 0
    for layer in squeezed.modules():
        if isinstance(layer, torch.nn.Linear):
            grid = layer.weight.data            # shape [judges, inputs]
            rows = grid.tolist()                # one list per judge
            new_rows = squeeze_per_channel(rows, bits)
            layer.weight.data = torch.tensor(new_rows, dtype=grid.dtype)
            grids += 1
    return squeezed, grids


def main():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from look_inside import MODEL_NAME
    from race import mean_absolute_error
    from whole_model import squeeze_model, answers, count_flips, REVIEWS

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Part 1: one grid. How much smaller is the rounding error?
    grid = model.distilbert.transformer.layer[0].attention.q_lin.weight.data
    rows = grid.tolist()
    flat = grid.flatten().tolist()
    print("\nOne grid (layer 0, q_lin): average error per weight")
    print("bits   per-tensor   per-channel")
    for bits in [8, 4, 2]:
        tensor_err = mean_absolute_error(flat, squeeze(flat, bits))
        channel_flat = [w for row in squeeze_per_channel(rows, bits) for w in row]
        channel_err = mean_absolute_error(flat, channel_flat)
        print(f"{bits:4d}   {tensor_err:10.5f}   {channel_err:11.5f}")

    # Part 2: the whole model. Do the 👍/👎 answers still flip?
    original = answers(model, tokenizer, REVIEWS)
    results = {}
    for bits in [8, 4, 2]:
        print(f"Squeezing all grids at {bits} bits (per-tensor, then per-channel)...")
        results[("tensor", bits)] = answers(squeeze_model(model, "linear", bits)[0], tokenizer, REVIEWS)
        results[("channel", bits)] = answers(squeeze_model_per_channel(model, bits)[0], tokenizer, REVIEWS)

    print(f"\nLinear: flips out of {len(REVIEWS)} reviews (vs the original model)\n")
    print("               8 bits   4 bits   2 bits")
    for kind in ["tensor", "channel"]:
        flips = [count_flips(original, results[(kind, b)]) for b in [8, 4, 2]]
        print(f"per-{kind:8s}   {flips[0]:6d}   {flips[1]:6d}   {flips[2]:6d}")

    print("\nreview" + " " * 47 + "orig  t8   t4   t2   c8   c4   c2")
    for i, review in enumerate(REVIEWS):
        row = [results[(k, b)][i] for k in ["tensor", "channel"] for b in [8, 4, 2]]
        print(f"{review[:50]:50s}   {original[i]}  " + "  ".join(row))


# Run this file to see per-tensor vs per-channel:  python per_channel.py
if __name__ == "__main__":
    main()
