# Day 22: Fake quantization. Let the model feel the rounding while it learns.
#
# Days 10-21 rounded the weights AFTER training. The model never got a chance
# to adjust. Sometimes a better set of buttons existed, and it never looked.
#
# Fake quantization keeps two copies of every weight:
#
#   hidden copy (smooth)  ->  round  ->  buttons  ->  compute the answer and the miss
#          ^                                                      |
#          +--------------- the nudge lands here -----------------+
#
# Forward:  round. The answer comes from the buttons, so the model feels them.
# Backward: pretend rounding isn't there (the "straight-through" trick).
#           Rounding is flat, so the true nudge would be 0 and nothing would learn.
#
# Tiny nudges pile up in the hidden copy until one crosses a halfway line,
# and then the button jumps. When training ends, we ship only the buttons.
# This is quantization-aware training (QAT). PyTorch's FakeQuantize does it.

import random
from zero_point import round_half_up

BUTTONS = [0, 1, 2, 3]       # weights may only sit here (2 bits, scale 1, to keep it simple)
NUM_WEIGHTS = 6
NUM_EXAMPLES = 40
STORY_SEED = 4               # the toy we tell the story with


def snap(weight):
    # Round to the nearest button, and stay inside 0..3.
    # Toy example: 1.4 -> 1,  1.5 -> 2,  -0.7 -> 0,  5.2 -> 3
    return min(max(round_half_up(weight), BUTTONS[0]), BUTTONS[-1])


def make_toy(seed):
    # A small made-up problem: 40 examples, 6 inputs each, one target number.
    # The "true" weights sit anywhere from 0 to 3, so most fall between buttons.
    # Inputs move together (like real features often do), plus a little noise.
    r = random.Random(seed)
    true_weights = [r.uniform(0, 3) for _ in range(NUM_WEIGHTS)]
    examples = []
    for _ in range(NUM_EXAMPLES):
        shared = r.uniform(-1, 1)
        inputs = [shared + r.uniform(-0.3, 0.3) for _ in range(NUM_WEIGHTS)]
        target = answer(true_weights, inputs)
        examples.append((inputs, target))
    return examples


def answer(weights, inputs):
    # One neuron: multiply each input by its weight and add them up.
    # Toy example: weights [1, 2], inputs [2, 2]  ->  2 + 4 = 6
    return sum(w * x for w, x in zip(weights, inputs))


def average_miss(weights, examples):
    # How far off is the neuron, on average? abs() stops misses from cancelling.
    misses = [abs(answer(weights, inputs) - target) for inputs, target in examples]
    return sum(misses) / len(misses)


def nudges(weights, examples):
    # Which way, and how hard, should each weight move? (Idea 1: big miss, big nudge.)
    # For one example: nudge for weight i = miss x input i.
    # Toy example: weights [2, 2], inputs [2, 2], target 6  ->  miss +2  ->  nudges [4, 4]
    # (positive means "too big, move down")
    total = [0.0] * len(weights)
    for inputs, target in examples:
        miss = answer(weights, inputs) - target
        for i in range(len(weights)):
            total[i] += miss * inputs[i]
    return [t / len(examples) for t in total]


def train_smooth(examples, steps=5000, rate=0.1):
    # Normal training: smooth weights, no buttons anywhere.
    weights = [0.0] * NUM_WEIGHTS
    for _ in range(steps):
        push = nudges(weights, examples)
        weights = [w - rate * p for w, p in zip(weights, push)]
    return weights


def train_buttons_only(start, examples, steps=3000, rate=0.003):
    # Way B: store only buttons. Every nudge is rounded straight away.
    # Tiny nudges get rounded back, so the weights never move.
    weights = [snap(w) for w in start]
    for step in range(steps):
        push = nudges(weights, examples)
        weights = [snap(w - rate * p) for w, p in zip(weights, push)]
    return weights


def train_fake_quant(start, examples, steps=3000, rate=0.003):
    # Way C: fake quantization with the straight-through trick.
    hidden = list(start)                       # the smooth notebook
    for step in range(steps):
        buttons = [snap(w) for w in hidden]    # forward: round, feel the buttons
        push = nudges(buttons, examples)       # the miss comes from the buttons
        shrink = 1 - step / steps              # nudges get smaller over time, so weights
                                               # stop flipping back and forth at the end
        hidden = [w - rate * shrink * p for w, p in zip(hidden, push)]
                                               # backward: the nudge goes straight to the
                                               # hidden copy, as if rounding weren't there
    return [snap(w) for w in hidden]           # ship only the buttons


def compare(seed):
    # Run all three ways on one toy. Returns the buttons and misses for each.
    examples = make_toy(seed)
    smooth = train_smooth(examples)
    a = [snap(w) for w in smooth]                       # A: round at the end
    b = train_buttons_only(a, examples)                 # B: buttons only
    c = train_fake_quant(smooth, examples)              # C: fake quant + straight-through
    return {
        "smooth": (smooth, average_miss(smooth, examples)),
        "A": (a, average_miss(a, examples)),
        "B": (b, average_miss(b, examples)),
        "C": (c, average_miss(c, examples)),
    }


def main():
    print(f"One neuron, {NUM_WEIGHTS} weights, buttons {BUTTONS}, {NUM_EXAMPLES} examples.\n")

    result = compare(STORY_SEED)
    smooth, smooth_miss = result["smooth"]
    print("Smooth training (no buttons):")
    print("  weights", [round(w, 2) for w in smooth], f" miss {smooth_miss:.3f}\n")

    labels = {"A": "A. round at the end   ",
              "B": "B. buttons only       ",
              "C": "C. fake quant + STE   "}
    for key in ("A", "B", "C"):
        buttons, miss = result[key]
        print(f"  {labels[key]} buttons {buttons}   miss {miss:.3f}")

    a_buttons = result["A"][0]
    c_buttons = result["C"][0]
    moved = [i for i in range(NUM_WEIGHTS) if a_buttons[i] != c_buttons[i]]
    print(f"\n  Fake quant moved {len(moved)} weight(s): "
          + ", ".join(f"w{i + 1} {a_buttons[i]} -> {c_buttons[i]}" for i in moved))

    # Is that one lucky toy? Try 20 different toys.
    wins = ties = losses = 0
    for seed in range(20):
        r = compare(seed)
        a_miss, c_miss = r["A"][1], r["C"][1]
        if c_miss < a_miss - 1e-9:
            wins += 1
        elif c_miss > a_miss + 1e-9:
            losses += 1
        else:
            ties += 1
    print(f"\nAcross 20 toys, fake quant vs round-at-the-end: "
          f"{wins} wins, {ties} ties, {losses} losses")


if __name__ == "__main__":
    main()
