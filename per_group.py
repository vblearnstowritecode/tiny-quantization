# Day 15: per-group. Cut every row into small groups, each with its own buttons.
#
# The whole idea in one line:
#   one outlier in row 3  ->  per-channel: all 768 weights in row 3 get wide steps
#                         ->  per-group:   only its group of 16 gets wide steps
#
# Cost: one extra scale per group. A 32-bit scale shared by 16 weights
# adds 32 / 16 = 2 bits per weight (like 16 friends splitting a $32 bill).

import torch

from look_inside import squeeze

GROUP_SIZE = 16


def cut_into_groups(row, group_size):
    # Toy example, group size 2:
    #   [1, 2, 3, 4, 5]  ->  [[1, 2], [3, 4], [5]]
    # If the row doesn't divide evenly, the last group is just shorter.
    groups = []
    for start in range(0, len(row), group_size):
        groups.append(row[start:start + group_size])
    return groups


def squeeze_row_per_group(row, bits, group_size=GROUP_SIZE):
    # Squeeze each group by itself, then glue the groups back into one row.
    # Toy example, 2 bits, group size 3:
    #   row [0.1, 0.2, 0.3, -5, 0, 5]
    #   per-channel: the -5..5 range sets one wide step -> 0.1, 0.2, 0.3 all become 0
    #   per-group:   [0.1, 0.2, 0.3] gets its own small step -> they stay 0.1, 0.2, 0.3
    new_row = []
    for group in cut_into_groups(row, group_size):
        new_row.extend(squeeze(group, bits))
    return new_row


def squeeze_per_group(rows, bits, group_size=GROUP_SIZE):
    return [squeeze_row_per_group(row, bits, group_size) for row in rows]


def extra_bits_per_weight(group_size, scale_bits=32):
    # Each weight's share of its group's scale. (Like splitting a dinner bill.)
    return scale_bits / group_size


def squeeze_model_per_group(model, bits, group_size=GROUP_SIZE):
    # Same as Day 14's squeeze_model_per_channel, but group by group inside each row.
    # Gives back a squeezed copy, plus how many grids we squeezed.
    import copy
    squeezed = copy.deepcopy(model)
    grids = 0
    for layer in squeezed.modules():
        if isinstance(layer, torch.nn.Linear):
            grid = layer.weight.data            # shape [rows, inputs]
            rows = grid.tolist()
            new_rows = squeeze_per_group(rows, bits, group_size)
            layer.weight.data = torch.tensor(new_rows, dtype=grid.dtype)
            grids += 1
    return squeezed, grids


def main():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from look_inside import MODEL_NAME
    from race import mean_absolute_error
    from per_channel import squeeze_per_channel
    from whole_model import answers, count_flips, REVIEWS

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Part 1: one grid. How much smaller is the rounding error?
    grid = model.distilbert.transformer.layer[0].attention.q_lin.weight.data
    rows = grid.tolist()
    flat = grid.flatten().tolist()
    print(f"\nOne grid (layer 0, q_lin): average error per weight (groups of {GROUP_SIZE})")
    print("bits   per-tensor   per-channel   per-group")
    for bits in [8, 4, 2]:
        tensor_err = mean_absolute_error(flat, squeeze(flat, bits))
        channel_flat = [w for row in squeeze_per_channel(rows, bits) for w in row]
        channel_err = mean_absolute_error(flat, channel_flat)
        group_flat = [w for row in squeeze_per_group(rows, bits) for w in row]
        group_err = mean_absolute_error(flat, group_flat)
        print(f"{bits:4d}   {tensor_err:10.5f}   {channel_err:11.5f}   {group_err:9.5f}")

    # Part 2: the whole model. Do the 👍/👎 answers still flip?
    # Per-tensor and per-channel take minutes, so we reuse their Day 12 and Day 14 results.
    original = answers(model, tokenizer, REVIEWS)
    results = {}
    for bits in [8, 4, 2]:
        print(f"Squeezing all grids at {bits} bits, per-group...")
        results[bits] = answers(squeeze_model_per_group(model, bits)[0], tokenizer, REVIEWS)

    print(f"\nLinear: flips out of {len(REVIEWS)} reviews (vs the original model)\n")
    print("               8 bits   4 bits   2 bits   extra bits per weight")
    print("per-tensor          0       10       10   ~0      (Day 12)")
    print(f"per-channel         0        1        6   {extra_bits_per_weight(768):.2f}    (Day 14)")
    flips = [count_flips(original, results[b]) for b in [8, 4, 2]]
    print(f"per-group      {flips[0]:6d}   {flips[1]:6d}   {flips[2]:6d}   "
          f"{extra_bits_per_weight(GROUP_SIZE):.2f}    (today)")

    print("\nreview" + " " * 47 + "orig  g8   g4   g2")
    for i, review in enumerate(REVIEWS):
        row = [results[b][i] for b in [8, 4, 2]]
        print(f"{review[:50]:50s}   {original[i]}  " + "  ".join(row))


# Run this file to see per-group results:  python per_group.py
if __name__ == "__main__":
    main()
