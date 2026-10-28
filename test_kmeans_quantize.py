# Checks for kmeans_quantize.py. Run with:  python test_kmeans_quantize.py
# Each test prints PASS or FAIL, so you can see exactly what works.

from kmeans_quantize import nearest_slot, assign, move_slots, kmeans


def close(a, b):
    # Computers store decimals slightly imperfectly, so we check "very close".
    return abs(a - b) < 1e-9


def test_nearest_clear_winner():
    # Kid at 4, carts at 0 and 3. Distances are 4 and 1, so the cart at 3 wins.
    return nearest_slot(4, [0, 3]) == 1


def test_nearest_tie_picks_first():
    # Kid at 2, carts at 0 and 4. Both are 2 away, so the first cart in the list wins.
    return nearest_slot(2, [0, 4]) == 0 and nearest_slot(2, [4, 0]) == 0


def test_nearest_negative():
    # Kid at -5, carts at -6 and 1. Distances are 1 and 6, so the cart at -6 wins.
    # Without abs(), -6 would look smaller than 1 and the wrong cart would win.
    return nearest_slot(-5, [-6, 1]) == 0


def test_nearest_on_top_of_slot():
    # Kid at 3, standing right on the cart at 3. Distance 0 always wins.
    return nearest_slot(3, [0, 3]) == 1


def test_assign_round_1():
    # Carts at 0 and 3. Only kid 0 picks cart A. Everyone else picks cart B.
    return assign([0, 4, 10, 16], [0, 3]) == [0, 1, 1, 1]


def test_assign_round_2_kid_4_switches():
    # Cart B moved to 10. Now kid 4 is closer to cart A (4 away vs 6 away).
    return assign([0, 4, 10, 16], [0, 10]) == [0, 0, 1, 1]


def test_move_round_1():
    # Cart A has {0}, so it stays at 0. Cart B has {4, 10, 16}, so it moves to 30 / 3 = 10.
    return move_slots([0, 4, 10, 16], [0, 1, 1, 1], [0, 3]) == [0, 10]


def test_move_round_2():
    # Cart A has {0, 4}, so it moves to 2. Cart B has {10, 16}, so it moves to 13.
    return move_slots([0, 4, 10, 16], [0, 0, 1, 1], [0, 10]) == [2, 13]


def test_move_lonely_slot_stays_put():
    # Nobody picks the cart at 100. It used to crash (0 / 0). Now it just stays at 100.
    return move_slots([0, 4, 10, 16], [0, 1, 1, 1], [0, 3, 100]) == [0, 10, 100]


def test_kmeans_kids_good_start():
    # Carts start at 0 and 3. After 3 rounds they settle at 2 and 13.
    slots, groups = kmeans([0, 4, 10, 16], [0, 3])
    return slots == [2, 13] and groups == [0, 0, 1, 1]


def test_kmeans_kids_bad_start_gets_stuck():
    # Carts start at 5 and 16. Kid 10 picks A (5 away vs 6), and nobody ever switches.
    # It still stops, but at a worse answer: A = 14 / 3, B = 16.
    slots, groups = kmeans([0, 4, 10, 16], [5, 16])
    return close(slots[0], 14 / 3) and slots[1] == 16 and groups == [0, 0, 0, 1]


def test_kmeans_our_weights():
    # Start from the 2-bit linear slots. K-means finds the slots we picked by eye.
    weights = [0.31, -0.82, 0.07, -0.45, 0.66]
    slots, groups = kmeans(weights, [-0.82, 0, 0.82])
    expected = [-0.635, 0.19, 0.66]
    return all(close(s, e) for s, e in zip(slots, expected)) and groups == [1, 0, 1, 0, 2]


def test_kmeans_no_weight_vanishes():
    # On Day 3, 0.07 and 0.31 became 0 at 2 bits. With K-means, no weight lands on 0.
    weights = [0.31, -0.82, 0.07, -0.45, 0.66]
    slots, groups = kmeans(weights, [-0.82, 0, 0.82])
    recovered = [slots[g] for g in groups]
    return all(r != 0 for r in recovered)


def test_kmeans_halves_worst_error():
    # Linear 2-bit: worst error 0.37 (-0.45 snaps to -0.82). K-means: worst error 0.185.
    weights = [0.31, -0.82, 0.07, -0.45, 0.66]
    slots, groups = kmeans(weights, [-0.82, 0, 0.82])
    worst = max(abs(w - slots[g]) for w, g in zip(weights, groups))
    return close(worst, 0.185)


if __name__ == "__main__":
    tests = [
        test_nearest_clear_winner,
        test_nearest_tie_picks_first,
        test_nearest_negative,
        test_nearest_on_top_of_slot,
        test_assign_round_1,
        test_assign_round_2_kid_4_switches,
        test_move_round_1,
        test_move_round_2,
        test_move_lonely_slot_stays_put,
        test_kmeans_kids_good_start,
        test_kmeans_kids_bad_start_gets_stuck,
        test_kmeans_our_weights,
        test_kmeans_no_weight_vanishes,
        test_kmeans_halves_worst_error,
    ]
    passed = 0
    for test in tests:
        try:
            ok = test()
        except Exception as e:
            ok = False
            print(f"FAIL  {test.__name__}  (crashed: {e})")
            continue
        print(("PASS  " if ok else "FAIL  ") + test.__name__)
        passed += ok
    print(f"\n{passed} of {len(tests)} tests passed")
