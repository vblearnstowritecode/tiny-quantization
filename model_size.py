# Day 8: why quantization lets a big model fit on your laptop.
#
# The whole idea in one line:
#   7 billion weights x 4 bytes (32-bit) = 28 GB  ->  too big for a 16 GB laptop
#   7 billion weights x 1 byte  (8-bit)  =  7 GB  ->  fits
#
# Same model, same weights, smaller suitcase.
# The price is the small rounding errors from Days 1-7.

LAPTOP_GB = 16


def size_in_gb(num_weights, bits):
    # 8 bits make 1 byte, and 1 billion bytes make 1 GB.
    # Toy example: 7 billion weights at 8 bits -> 7 billion bytes -> 7 GB
    num_bytes = num_weights * bits / 8
    return num_bytes / 1_000_000_000


def fits_on_laptop(size_gb):
    return size_gb <= LAPTOP_GB


if __name__ == "__main__":
    models = {"1B": 1_000_000_000, "7B": 7_000_000_000, "70B": 70_000_000_000}

    print(f"Does it fit on a {LAPTOP_GB} GB laptop?\n")
    for name, num_weights in models.items():
        print(f"{name} model")
        for bits in (32, 16, 8, 4):
            size = size_in_gb(num_weights, bits)
            mark = "fits" if fits_on_laptop(size) else "too big"
            print(f"  {bits:>2}-bit: {size:6.1f} GB   {mark}")
        print()
