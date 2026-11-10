# Checks for real_formats.py. Run with:  python test_real_formats.py
# Pure arithmetic, so it runs instantly.

from real_formats import bits_per_weight, FORMATS


def close(a, b, tol=0.001):
    return abs(a - b) < tol


def bill(name):
    weight_bits, scales = FORMATS[name]
    return bits_per_weight(weight_bits, scales)


# --- The basic arithmetic ---

def test_no_scales_is_just_weight_bits():
    return bits_per_weight(4, []) == 4


def test_one_scale_dinner_bill():
    # A 32-bit scale shared by 16 weights adds 2 bits.
    return bits_per_weight(4, [(32, 16)]) == 6


def test_scale_shared_by_one_weight():
    # Edge case: a scale for every single weight doubles the bill at 32 + 32.
    return bits_per_weight(32, [(32, 1)]) == 64


def test_bigger_groups_cheaper():
    return bits_per_weight(4, [(32, 64)]) < bits_per_weight(4, [(32, 16)])


def test_other_weight_bits():
    # 2-bit and 8-bit weights, same 8-bit scale per 16.
    return (bits_per_weight(2, [(8, 16)]) == 2.5 and
            bits_per_weight(8, [(8, 16)]) == 8.5)


# --- The four formats ---

def test_day15_bill():
    return bill("Day 15 (ours, 1 level)") == 6


def test_day16_bill():
    return close(bill("Day 16 (ours, 2 levels)"), 4.5)


def test_gguf_q4k_bill():
    # 4 + 6/32 + 6/32 + 16/256 + 16/256 = 4.5 exactly.
    return bill("GGUF Q4_K") == 4.5


def test_gguf_matches_real_block_size():
    # A real Q4_K super-block is 144 bytes for 256 weights.
    return bill("GGUF Q4_K") * 256 / 8 == 144


def test_nvfp4_bill():
    return close(bill("NVFP4"), 4.5)


def test_nvfp4_same_as_day16():
    # The point of today: NVIDIA's layout is our Day 16 layout.
    return FORMATS["NVFP4"] == FORMATS["Day 16 (ours, 2 levels)"]


def test_two_levels_beat_one():
    return bill("Day 16 (ours, 2 levels)") < bill("Day 15 (ours, 1 level)")


def test_grid_scale_almost_free():
    # The 32-bit grid scale adds less than 0.001 bits per weight.
    return bill("NVFP4") - 4.5 < 0.001


if __name__ == "__main__":
    tests = [
        test_no_scales_is_just_weight_bits,
        test_one_scale_dinner_bill,
        test_scale_shared_by_one_weight,
        test_bigger_groups_cheaper,
        test_other_weight_bits,
        test_day15_bill,
        test_day16_bill,
        test_gguf_q4k_bill,
        test_gguf_matches_real_block_size,
        test_nvfp4_bill,
        test_nvfp4_same_as_day16,
        test_two_levels_beat_one,
        test_grid_scale_almost_free,
    ]
    passed = 0
    for t in tests:
        ok = t()
        print(("PASS  " if ok else "FAIL  ") + t.__name__)
        passed += ok
    print(f"\n{passed} of {len(tests)} tests passed")
