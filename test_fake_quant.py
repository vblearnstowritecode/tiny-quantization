# Checks for fake_quant.py. Run with:  python test_fake_quant.py
# No model download. Takes about 15 seconds (the 20-toy test trains 20 times).

from fake_quant import (snap, make_toy, answer, average_miss, nudges, train_smooth,
                        train_buttons_only, train_fake_quant, compare, BUTTONS, STORY_SEED)


# --- Snapping to a button ---

def test_snap_toy():
    return [snap(1.4), snap(1.5), snap(-0.7), snap(5.2)] == [1, 2, 0, 3]


def test_snap_buttons_stay_put():
    return [snap(b) for b in BUTTONS] == BUTTONS


def test_snap_just_below_halfway():
    return snap(0.49) == 0 and snap(2.49) == 2


def test_snap_rounding_is_flat():
    # Idea 4: move the weight a little, the button doesn't change.
    # So the true nudge through rounding is 0. That's why we need the trick.
    return snap(1.6) == snap(1.61) == snap(1.7)


# --- The neuron and its miss ---

def test_answer_toy():
    return answer([1, 2], [2, 2]) == 6


def test_answer_all_zero_weights():
    return answer([0, 0, 0], [5, -3, 2]) == 0


def test_average_miss_perfect_is_zero():
    return average_miss([1, 2], [([2, 2], 6), ([1, 1], 3)]) == 0


def test_average_miss_does_not_cancel():
    # One miss of +1 and one of -1 average to 1, not 0.
    return average_miss([1], [([1], 0), ([1], 2)]) == 1


# --- Nudges ---

def test_nudges_toy():
    # Lesson toy: weights [2, 2], inputs [2, 2], target 6 -> miss +2 -> nudges [4, 4].
    return nudges([2, 2], [([2, 2], 6)]) == [4, 4]


def test_nudges_zero_when_perfect():
    return nudges([1, 2], [([2, 2], 6)]) == [0, 0]


def test_nudges_negative_when_too_small():
    # Answer 2, target 6: too small, so the nudge says "move up" (negative).
    return all(p < 0 for p in nudges([1, 0], [([2, 2], 6)]))


def test_bigger_miss_bigger_nudge():
    small = nudges([2.5], [([1], 2)])[0]     # miss 0.5
    big = nudges([4.0], [([1], 2)])[0]       # miss 2
    return big > small > 0


# --- The toy problem ---

def test_toy_is_repeatable():
    return make_toy(7) == make_toy(7)


def test_toy_has_right_shape():
    examples = make_toy(0)
    return len(examples) == 40 and all(len(inputs) == 6 for inputs, _ in examples)


def test_smooth_training_fits_the_toy():
    examples = make_toy(STORY_SEED)
    return average_miss(train_smooth(examples), examples) < 0.001


# --- Buttons only (way B): the nudges vanish ---

def test_buttons_only_is_stuck():
    # Start at 1.6 -> button 2. Target wants 1. Each nudge (0.003) rounds back to 2.
    return train_buttons_only([1.6], [([1], 1.0)]) == [2]


def test_buttons_only_matches_a_on_story_toy():
    r = compare(STORY_SEED)
    return r["B"][0] == r["A"][0]


# --- Fake quant (way C): the hidden copy collects the nudges ---

def test_fake_quant_button_jumps():
    # Same start as above. The hidden copy creeps down past 1.5, so the button jumps to 1.
    return train_fake_quant([1.6], [([1], 1.0)]) == [1]


def test_fake_quant_stays_on_a_perfect_button():
    # Already exact: the miss is 0, so nothing moves.
    return train_fake_quant([1.0, 2.0], [([2, 2], 6), ([1, 3], 7)]) == [1, 2]


def test_fake_quant_all_zero_targets():
    return train_fake_quant([0.2, 0.3], [([1, 1], 0), ([2, -1], 0)]) == [0, 0]


def test_fake_quant_only_ships_buttons():
    r = compare(0)
    return all(w in BUTTONS for w in r["C"][0])


def test_fake_quant_wins_story_toy():
    # The problem fake quant fixes: round-at-the-end misses 0.596, fake quant 0.111.
    r = compare(STORY_SEED)
    return r["C"][1] < r["A"][1] / 3


def test_story_toy_moves_one_weight():
    r = compare(STORY_SEED)
    moved = [i for i in range(6) if r["A"][0][i] != r["C"][0][i]]
    return moved == [3]                       # w4: 0 -> 1


def test_fake_quant_wins_more_than_it_loses():
    # Honest check over 20 toys: it doesn't always win, but it wins far more often.
    wins = losses = 0
    for seed in range(20):
        r = compare(seed)
        if r["C"][1] < r["A"][1] - 1e-9:
            wins += 1
        elif r["C"][1] > r["A"][1] + 1e-9:
            losses += 1
    return wins == 9 and losses == 2


# --- Run them all ---

if __name__ == "__main__":
    tests = [(name, fn) for name, fn in list(globals().items()) if name.startswith("test_")]
    passed = 0
    for name, fn in tests:
        ok = fn()
        print(("PASS  " if ok else "FAIL  ") + name)
        passed += ok
    print(f"\n{passed} of {len(tests)} tests passed")
