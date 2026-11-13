# Day 20: Spikes in activations, and FP4's wasted mirror.
#
# Weights are fixed. Activations are the numbers a token carries between layers,
# new for every sentence. A few channels (columns) are huge for almost every token:
#
#              ch 0    ch 1   ...   ch 308   ...
#   "i"        0.3    -0.5    ...   -8.4     ...
#   "loved"    0.1     0.4    ...   -6.5     ...    <- channel 308 = one spiky judge
#
# We cut each token's row into groups of 16 channels. The group holding the spike
# gets a wide scale, so its small numbers snap to 0. That hurts linear AND FP4.
#
# FP4 also builds its stops as a mirror (same on + and - sides). The spike here is
# one-sided (negative), so FP4's + side stops sit empty. Linear with a zero point
# fits min..max, so it wastes nothing.
#
#   FP4 on a group from -6.5 to +0.5:   ●●●●●●●●●○○○○○○   (○ = empty stop)
#   linear on the same group:           ●●●●●●●●●●●●●●●●  (all inside -6.5..0.5)

from look_inside import squeeze, MODEL_NAME
from fp4 import squeeze_fp4
from per_group import cut_into_groups, GROUP_SIZE
from race import mean_absolute_error

REVIEW = "I loved every minute of this movie."
STOP = 3   # the activations passed from layer 3 to layer 4 (spikes are strong here)


def get_activations(model, tokenizer, review, stop=STOP):
    # Run the model and copy the numbers passed between layers at one stop.
    # Gives back a table: one row per token, 768 numbers (channels) per row.
    import torch
    inputs = tokenizer(review, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    rows = out.hidden_states[stop][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    return rows, tokens


def find_spike_channel(rows):
    # The channel (column) that is big for EVERY token, not just one.
    # So we score each column by its average size (ignoring sign).
    # Toy example:  [[ 1, -9, 2],
    #                [ 0, -8, 1],
    #                [20, -7, 1]]   ->  channel 1 (average 8; channel 0 has one 20, average 7)
    best_channel = 0
    best_size = 0
    for channel in range(len(rows[0])):
        column = [row[channel] for row in rows]
        average_size = sum(abs(v) for v in column) / len(column)
        if average_size > best_size:
            best_size = average_size
            best_channel = channel
    return best_channel


def split_calm_and_spike(rows, spike_channel, group_size=GROUP_SIZE):
    # Cut every row into groups. Sort each group into one of two piles:
    # "spike" if it holds the spike channel, "calm" if not.
    # Toy example, group size 2, spike channel 1:
    #   [[1, -9, 2, 3]]  ->  spike [[1, -9]],  calm [[2, 3]]
    spike_group_number = spike_channel // group_size
    calm, spike = [], []
    for row in rows:
        for number, group in enumerate(cut_into_groups(row, group_size)):
            if number == spike_group_number:
                spike.append(group)
            else:
                calm.append(group)
    return calm, spike


def errors(groups):
    # Squeeze every group by itself (4 bits), both ways. Give back (linear MAE, FP4 MAE).
    original, linear, fp4 = [], [], []
    for group in groups:
        original.extend(group)
        linear.extend(squeeze(group, 4))
        fp4.extend(squeeze_fp4(group))
    return mean_absolute_error(original, linear), mean_absolute_error(original, fp4)


def stops_used(squeezed):
    # How many different stops did a squeezed group actually land on?
    # Toy example: [0, 0, -2, -2, -6]  ->  3 stops (0, -2, -6)
    return len(set(round(s, 6) for s in squeezed))


def main():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    rows, tokens = get_activations(model, tokenizer, REVIEW)
    spike = find_spike_channel(rows)
    print(f'\nReview: "{REVIEW}"')
    print(f"Activations between layers {STOP} and {STOP + 1}: "
          f"{len(rows)} tokens x {len(rows[0])} channels")

    print(f"\nSpike channel: {spike}. Its value for every token:")
    for token, row in zip(tokens, rows):
        print(f"  {token:>8}  {row[spike]:6.1f}")

    calm, spiky = split_calm_and_spike(rows, spike)
    calm_lin, calm_fp4 = errors(calm)
    spike_lin, spike_fp4 = errors(spiky)
    print(f"\n4 bits, groups of {GROUP_SIZE}. Error (MAE), lower is better:")
    print("                  linear      FP4")
    print(f"calm groups     {calm_lin:8.4f} {calm_fp4:8.4f}   ({len(calm)} groups)")
    print(f"spike groups    {spike_lin:8.4f} {spike_fp4:8.4f}   ({len(spiky)} groups)")

    # Look at one real spike group up close.
    token = 3
    start = (spike // GROUP_SIZE) * GROUP_SIZE
    group = rows[token][start:start + GROUP_SIZE]
    lin = squeeze(group, 4)
    fp4 = squeeze_fp4(group)
    print(f'\nOne spike group: token "{tokens[token]}", channels {start}..{start + GROUP_SIZE - 1}')
    print("  original  linear     FP4")
    for o, l, f in zip(group, lin, fp4):
        print(f"  {o:8.2f}  {l:6.2f}  {f:6.2f}")
    print(f"range {min(group):.2f} .. {max(group):.2f}")
    print(f"stops used: linear {stops_used(lin)}, FP4 {stops_used(fp4)} "
          f"(FP4 has 15, but most + side stops sit empty)")


# Run this file to see it:  python activations.py
if __name__ == "__main__":
    main()
