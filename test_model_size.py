# Checks for model_size.py. Run with:  python test_model_size.py
# Each test prints PASS or FAIL, so you can see exactly what works.

from model_size import size_in_gb, fits_on_laptop


def test_7b_at_32_bits():
    # 4 bytes per weight -> 28 GB
    return size_in_gb(7_000_000_000, 32) == 28


def test_7b_at_8_bits():
    # Avi's answer: 1 byte per weight -> 7 GB
    return size_in_gb(7_000_000_000, 8) == 7


def test_7b_at_4_bits():
    # Half a byte per weight -> 3.5 GB
    return size_in_gb(7_000_000_000, 4) == 3.5


def test_halving_bits_halves_size():
    # Every time bits halve, the suitcase halves too
    for bits in (32, 16, 8, 4):
        if size_in_gb(1_000_000_000, bits // 2) != size_in_gb(1_000_000_000, bits) / 2:
            return False
    return True


def test_zero_weights():
    # Edge case: no weights, no memory
    return size_in_gb(0, 8) == 0


def test_the_problem_7b_too_big_at_32_bits():
    # The problem quantization fixes: 28 GB does not fit in 16 GB
    return not fits_on_laptop(size_in_gb(7_000_000_000, 32))


def test_the_fix_7b_fits_at_8_bits():
    return fits_on_laptop(size_in_gb(7_000_000_000, 8))


def test_exactly_16_gb_fits():
    # Edge case: right at the limit still counts as fitting
    return fits_on_laptop(16)


def test_70b_too_big_even_at_4_bits():
    # 35 GB: some models are too big for a laptop no matter what
    return size_in_gb(70_000_000_000, 4) == 35 and not fits_on_laptop(35)


if __name__ == "__main__":
    tests = [test_7b_at_32_bits, test_7b_at_8_bits, test_7b_at_4_bits,
             test_halving_bits_halves_size, test_zero_weights,
             test_the_problem_7b_too_big_at_32_bits, test_the_fix_7b_fits_at_8_bits,
             test_exactly_16_gb_fits, test_70b_too_big_even_at_4_bits]
    passed = 0
    for t in tests:
        ok = t()
        passed += ok
        print(("PASS  " if ok else "FAIL  ") + t.__name__)
    print(f"\n{passed} of {len(tests)} tests passed")
