# Day 9: we shrank the model. Does it still give the right answer?
#
# The whole idea in one line:
#   each weight gets a small rounding error  ->  the errors add up in the answer
#   some push up, some push down, so they often partly cancel
#
# We test the smallest "model" there is: one neuron.
# It multiplies each input by a weight and adds everything up.

from shift_quantize import get_start_and_step, quantize, dequantize


def neuron(inputs, weights):
    # Multiply and add, like a grocery bill: amount x price, then the total.
    # Toy example: inputs [1, 2], weights [0.5, 0.25]  ->  1*0.5 + 2*0.25 = 1.0
    total = 0
    for x, w in zip(inputs, weights):
        total += x * w
    return total


def shrink(weights, bits):
    # Quantize the weights to a few bits, then convert them back (Day 5).
    # The weights come back slightly off. That's the rounding error.
    start, step = get_start_and_step(weights, bits)
    slots = quantize(weights, start, step)
    return dequantize(slots, start, step)


# Run this file to see the drift:  python neuron_check.py
if __name__ == "__main__":
    inputs = [1, 2, 3]
    weights = [0.21, -0.48, 0.73]

    real_answer = neuron(inputs, weights)

    print(f"inputs  = {inputs}")
    print(f"weights = {weights}\n")
    print("bits   answer   off by")
    print(f"full   {real_answer:6.3f}   -")
    for bits in (8, 4, 2):
        answer = neuron(inputs, shrink(weights, bits))
        print(f"{bits:>4}   {answer:6.3f}   {abs(answer - real_answer):.3f}")
