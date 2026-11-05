# Checks for whole_model.py. Run with:  python test_whole_model.py
# These use made-up weights, so they run fast and need no download.

import random

import torch

from whole_model import kmeans_squeeze_sampled, squeeze_model, count_flips, answers, REVIEWS
from race import mean_absolute_error, kmeans_squeeze
from look_inside import squeeze


def bell_weights(count, seed):
    # Made-up weights shaped like a real layer: a big crowd near 0, a few far out.
    rng = random.Random(seed)
    return [rng.gauss(0, 0.05) for _ in range(count)] + [0.7, -0.6]


# --- Step 1: K-means with a poll ---

def test_sample_close_to_full():
    # The poll (5,000 of 100,000) should be within 5% of asking everyone.
    w = bell_weights(100_000, seed=1)
    full = mean_absolute_error(w, kmeans_squeeze(w, 4))
    polled = mean_absolute_error(w, kmeans_squeeze_sampled(w, 4, sample_size=5_000).tolist())
    return polled <= full * 1.05


def test_sample_beats_linear():
    # Even from a poll, K-means should still beat linear on a bell shape.
    w = bell_weights(100_000, seed=2)
    linear = mean_absolute_error(w, squeeze(w, 4))
    polled = mean_absolute_error(w, kmeans_squeeze_sampled(w, 4, sample_size=5_000).tolist())
    return polled < linear


def test_every_weight_gets_a_cart():
    # Weights left out of the poll must still be snapped. Same length, at most 2^bits values.
    w = bell_weights(20_000, seed=3)
    out = kmeans_squeeze_sampled(w, 2, sample_size=1_000)
    return len(out) == len(w) and len(set(out.tolist())) <= 4


def test_same_seed_same_answer():
    # A fixed seed means the same poll, so the same carts every run.
    # "Same" means very close: scikit-learn adds on several CPU cores, and the
    # adding order can change the last digit (about 1e-17).
    w = bell_weights(20_000, seed=4)
    a = kmeans_squeeze_sampled(w, 4, sample_size=1_000, seed=7)
    b = kmeans_squeeze_sampled(w, 4, sample_size=1_000, seed=7)
    return abs(a - b).max() < 1e-9


def test_small_grid_uses_everyone():
    # Fewer weights than the sample size: poll everyone, same as Day 11.
    w = bell_weights(500, seed=5)
    polled = kmeans_squeeze_sampled(w, 2, sample_size=50_000).tolist()
    full = kmeans_squeeze(w, 2)
    return all(abs(p - f) < 1e-6 for p, f in zip(polled, full))


def test_few_weights_kept_exact():
    w = [0.1, -0.2, 0.3]
    return kmeans_squeeze_sampled(w, 2).tolist() == w


def test_all_equal():
    return kmeans_squeeze_sampled([5.0] * 50, 2).tolist() == [5.0] * 50


def test_all_zero():
    return kmeans_squeeze_sampled([0.0] * 50, 4).tolist() == [0.0] * 50


def test_more_bits_less_error():
    w = bell_weights(20_000, seed=6)
    e = {b: mean_absolute_error(w, kmeans_squeeze_sampled(w, b, sample_size=2_000).tolist())
         for b in [2, 4, 8]}
    return e[8] < e[4] < e[2]


# --- Step 2: squeeze a copy of the whole model ---

def tiny_model():
    # A made-up model: 2 grids with very different ranges, plus a ReLU in between.
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(40, 30),
        torch.nn.ReLU(),
        torch.nn.Linear(30, 2),
    )
    with torch.no_grad():
        model[0].weight.mul_(10)     # wide grid
        model[2].weight.mul_(0.1)    # tight grid (like the real classifier)
    return model


def all_grids(model):
    return [m.weight for m in model.modules() if isinstance(m, torch.nn.Linear)]


def test_original_untouched():
    model = tiny_model()
    before = [g.clone() for g in all_grids(model)]
    squeeze_model(model, "linear", 2)
    squeeze_model(model, "kmeans", 2)
    return all(torch.equal(b, g) for b, g in zip(before, all_grids(model)))


def test_counts_every_grid():
    _, grids = squeeze_model(tiny_model(), "linear", 4)
    return grids == 2


def test_each_grid_has_few_values():
    # Every grid ends up with at most 2^bits different values.
    for method in ["linear", "kmeans"]:
        for bits in [2, 4]:
            squeezed, _ = squeeze_model(tiny_model(), method, bits)
            for g in all_grids(squeezed):
                if len(torch.unique(g)) > 2 ** bits:
                    return False
    return True


def test_shapes_kept():
    model = tiny_model()
    squeezed, _ = squeeze_model(model, "kmeans", 2)
    return [g.shape for g in all_grids(model)] == [g.shape for g in all_grids(squeezed)]


def test_biases_untouched():
    # We only squeeze the grids. The small bias lists stay exact.
    model = tiny_model()
    squeezed, _ = squeeze_model(model, "linear", 2)
    return torch.equal(model[0].bias, squeezed[0].bias)


def test_tight_grid_survives():
    # Idea 1: each grid gets its own buttons. The tight grid must NOT vanish to 0,
    # even though the other grid is 100x wider.
    squeezed, _ = squeeze_model(tiny_model(), "linear", 4)
    tight = squeezed[2].weight
    return (tight != 0).float().mean().item() > 0.5


def test_8_bits_answers_close():
    # At 8 bits, the squeezed model's outputs should barely move.
    model = tiny_model()
    x = torch.randn(5, 40, generator=torch.Generator().manual_seed(1))
    for method in ["linear", "kmeans"]:
        squeezed, _ = squeeze_model(model, method, 8)
        drift = (model(x) - squeezed(x)).abs().max().item()
        if drift > 0.05:
            return False
    return True


def test_unknown_method_rejected():
    try:
        squeeze_model(tiny_model(), "magic", 4)
        return False
    except ValueError:
        return True


# --- Step 3: count the flips ---

def test_flips_toy():
    return count_flips(["pos", "neg", "pos"], ["pos", "pos", "pos"]) == 1


def test_no_flips_when_same():
    return count_flips(["pos", "neg"], ["pos", "neg"]) == 0


def test_all_flipped():
    return count_flips(["pos", "neg", "neg"], ["neg", "pos", "pos"]) == 3


def test_twenty_reviews():
    return len(REVIEWS) == 20 and len(set(REVIEWS)) == 20


class FakeOutput:
    def __init__(self, logits):
        self.logits = logits


class FakeModel(torch.nn.Module):
    # Says 👍 when the review has the word "good" in it. Two scores: [👎, 👍].
    def forward(self, has_good):
        return FakeOutput(torch.stack([1 - has_good, has_good], dim=1))


def fake_tokenizer(reviews, padding, return_tensors):
    return {"has_good": torch.tensor([1.0 if "good" in r else 0.0 for r in reviews])}


def test_answers_reads_the_right_score():
    # Score [👎, 👍]: the second number is 👍. Mixing them up would flip everything.
    out = answers(FakeModel(), fake_tokenizer, ["good film", "bad film"])
    return out == ["pos", "neg"]


def test_stuck_model_flips_all_negatives():
    # The broken thermometer: a model that always says pos.
    # On 10 pos + 10 neg reviews, it flips exactly the 10 negatives.
    original = ["pos"] * 10 + ["neg"] * 10
    stuck = ["pos"] * 20
    return count_flips(original, stuck) == 10


if __name__ == "__main__":
    tests = [
        test_sample_close_to_full,
        test_sample_beats_linear,
        test_every_weight_gets_a_cart,
        test_same_seed_same_answer,
        test_small_grid_uses_everyone,
        test_few_weights_kept_exact,
        test_all_equal,
        test_all_zero,
        test_more_bits_less_error,
        test_original_untouched,
        test_counts_every_grid,
        test_each_grid_has_few_values,
        test_shapes_kept,
        test_biases_untouched,
        test_tight_grid_survives,
        test_8_bits_answers_close,
        test_unknown_method_rejected,
        test_flips_toy,
        test_no_flips_when_same,
        test_all_flipped,
        test_twenty_reviews,
        test_answers_reads_the_right_score,
        test_stuck_model_flips_all_negatives,
    ]
    passed = 0
    for t in tests:
        ok = t()
        print(("PASS  " if ok else "FAIL  ") + t.__name__)
        passed += ok
    print(f"\n{passed} of {len(tests)} tests passed")
