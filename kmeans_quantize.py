# K-means quantization: let the slots move to where the weights actually are.
#
# Picture ice cream carts (slots) on a beach, and kids (weights) standing around.
#   Rule 1: each kid walks to the nearest cart.
#   Rule 2: each cart moves to the middle of its kids.
# Repeat until no kid switches carts.


def nearest_slot(weight, slots):
    # Which slot is closest to this weight? Gives back its place in the list.
    # Toy example: weight 4, slots [0, 3]  ->  distances [4, 1]  ->  1 (the slot at 3)
    distances = [abs(weight - s) for s in slots]
    smallest = min(distances)
    return distances.index(smallest)


def assign(weights, slots):
    # Rule 1: every weight picks its nearest slot.
    # Toy example: weights [0, 4, 10, 16], slots [0, 3]  ->  [0, 1, 1, 1]
    return [nearest_slot(w, slots) for w in weights]


def move_slots(weights, groups, slots):
    # Rule 2: every slot moves to the average of the weights that picked it.
    # Toy example: weights [0, 4, 10, 16], groups [0, 0, 1, 1]  ->  [2, 13]
    # A slot that nobody picked stays where it is (we can't average nothing).
    new_slots = []
    for cart in range(len(slots)):
        my_kids = [w for w, g in zip(weights, groups) if g == cart]
        if len(my_kids) == 0:
            new_slots.append(slots[cart])
        else:
            new_slots.append(sum(my_kids) / len(my_kids))
    return new_slots


def kmeans(weights, slots):
    # Repeat Rule 2 and Rule 1 until no weight switches slots.
    # Toy example: weights [0, 4, 10, 16], starting slots [0, 3]  ->  slots [2, 13]
    # Gives back the final slots, and which slot each weight picked.
    groups = assign(weights, slots)
    while True:
        slots = move_slots(weights, groups, slots)
        new_groups = assign(weights, slots)
        if new_groups == groups:
            return slots, groups
        groups = new_groups


# Run this file to try it:  python kmeans_quantize.py
if __name__ == "__main__":
    weights = [0.31, -0.82, 0.07, -0.45, 0.66]
    start = [-0.82, 0, 0.82]  # the 2-bit linear slots
    slots, groups = kmeans(weights, start)

    print("codebook:", [round(s, 3) for s in slots], "\n")
    print("weight   label   slot     error")
    for w, g in zip(weights, groups):
        print(f"{w:6.2f}   {g:5d}   {slots[g]:6.3f}   {abs(w - slots[g]):.3f}")
