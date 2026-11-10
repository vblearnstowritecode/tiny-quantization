# Day 17: GGUF and NVFP4 are our Day 16 design, built for real.
#
# This script quantizes nothing. It only adds up the storage bill.
#
# A format = bits for each weight + a list of scales.
# Each scale is (its bits, how many weights share it).
# Like a dinner bill, each scale's bits are split across the weights that share it:
#   an 8-bit scale shared by 16 weights  ->  8 / 16 = 0.5 extra bits per weight

GRID_WEIGHTS = 768 * 768   # one q_lin grid from our model: 589,824 weights


def bits_per_weight(weight_bits, scales):
    # Toy example (Day 16): 4-bit weights, an 8-bit scale per 16, a 32-bit scale per grid
    #   4 + 8/16 + 32/589824  =  4 + 0.5 + 0.00005  =  about 4.5
    total = weight_bits
    for scale_bits, shared_by in scales:
        total = total + scale_bits / shared_by
    return total


FORMATS = {
    # Day 15: one 32-bit scale per group of 16.
    "Day 15 (ours, 1 level)": (4, [(32, 16)]),

    # Day 16: an 8-bit scale per group of 16, plus one 32-bit scale per grid.
    "Day 16 (ours, 2 levels)": (4, [(8, 16), (32, GRID_WEIGHTS)]),

    # GGUF Q4_K: a "super-block" of 256 weights, cut into 8 blocks of 32.
    # Each block has a 6-bit scale and a 6-bit min (min = where the staircase starts).
    # The super-block has a 16-bit scale for the scales, and a 16-bit one for the mins.
    "GGUF Q4_K": (4, [(6, 32), (6, 32), (16, 256), (16, 256)]),

    # NVFP4 (NVIDIA): an 8-bit scale per block of 16, plus one 32-bit scale per grid.
    "NVFP4": (4, [(8, 16), (32, GRID_WEIGHTS)]),
}


if __name__ == "__main__":
    print("Storage bill, in bits per weight (4-bit weights)\n")
    for name, (weight_bits, scales) in FORMATS.items():
        bill = bits_per_weight(weight_bits, scales)
        print(f"  {name:<25} {bill:.2f} bits")
    print("\nGGUF and NVFP4 both use two levels of scales, like our Day 16.")
