# tiny-quantization

Quantization, worked out by hand with small numbers. One idea per day.

I'm learning from [MIT EfficientML, Lecture 5](https://www.youtube.com/watch?v=Qg_3N8pdK9s).
This repo is my notes in code form, written for someone new to the topic.

---

## Day 1: Linear quantization

**The problem.** A model stores each weight as a 32-bit decimal number.
That takes a lot of memory. Moving memory around is slow and costly.

**The idea.** Store each weight in only 8 bits instead.
8 bits give us 256 whole-number "slots" (-128 to 127). We use -127 to 127.
That's 4× less memory.

**The catch.** A weight like `0.23` is not a whole number. So which slot does it go into?

### Why we can't just round

Most weights are small, between -1 and 1. Rounding them squashes almost everything to 0:

```
weights :  0.31  -0.82   0.07  -0.45   0.66
rounded :  0     -1      0      0      1
```

The model loses nearly all its information.

### The fix: stretch, then round

Let's use a toy box with only 11 slots, from -5 to 5. Our weights are:

```
0.2   -0.5   0.4   0.1
```

**Step 1. Find the scale.** The biggest weight (ignoring the minus sign) is 0.5.
We want it to land on the top slot, 5. So each slot step is worth:

```
scale = 0.5 / 5 = 0.1

real :  0    0.1   0.2   0.3   0.4   0.5
slots:  0 ── 1 ── 2 ── 3 ── 4 ── 5
```

**Step 2. Put a weight into a slot.** Divide by the scale, then round.

```
0.23 / 0.1 = 2.3   →   round   →   slot 2
```

**Step 3. Get it back.** Multiply the slot by the scale.

```
slot 2 × 0.1 = 0.2
```

We put in `0.23` and got back `0.2`. We lost `0.03`. That loss is the **rounding error**.

### How bad can the error get?

Think of a lift that only stops at whole floors. You want to reach height 3.2.
You get out at the nearest floor, 3, and miss by 0.2.

The worst case is a target exactly halfway, like 3.5. You miss by 0.5, half a floor. Never more.

Same with slots. Between slot 2 (0.2) and slot 3 (0.3), the worst weight is `0.25`.
It loses `0.05`, which is **half a step**.

> **The error is never more than half a step.**
> Smaller steps mean smaller errors.

### The whole algorithm

```
scale     = biggest weight / top slot
slot      = round(weight / scale)
recovered = slot × scale
```

Real int8 is the same thing with top slot 127 instead of 5.

---

## Run it

No libraries needed, just Python 3.

```
python quantize.py         # see a round trip on 5 weights
python compare_bits.py     # compare 8, 4 and 2 bits
python zero_weights.py     # see which weights become zero
python test_quantize.py    # run all the checks
```

Output of `python quantize.py`:

```
scale = 0.00646

original   slot   recovered   error
  0.3100     48      0.3099   0.0001
 -0.8200   -127     -0.8200   0.0000
  0.0700     11      0.0710   0.0010
 -0.4500    -70     -0.4520   0.0020
  0.6600    102      0.6586   0.0014
```

Every error stays under half a step (0.00323).

---

## Day 2: Fewer bits

8 bits worked well. What if we use 4 bits, or just 2? Less memory again. What does it cost?

### How many slots?

`n` bits give `2^n` slots. We split them around zero and skip the lone bottom slot,
so both sides mirror each other. The top slot is:

```
top slot = 2^(bits - 1) - 1

8 bits  →  127     slots -127 … 127
4 bits  →  7       slots   -7 … 7
2 bits  →  1       slots   -1 … 1
```

### Think of a staircase

The biggest weight is a platform. The top slot is the number of steps up to it.
The scale is the height of one step.

```
scale = biggest weight / top slot

4 bits: 7 steps            2 bits: 1 step

            ┌─ 0.82                   ┌─ 0.82
          ┌─┘                         │
        ┌─┘                           │
      ┌─┘                             │
    ┌─┘                               │
  ┌─┘                                 │
┌─┘                                   │
0                                     0

step = 0.82 / 7 = 0.117      step = 0.82 / 1 = 0.82
```

Fewer steps means taller steps. Every weight must stand on its nearest step.
From Day 1, the worst miss is half a step. So taller steps mean bigger misses.

### The result

Run `python compare_bits.py`:

```
bits   slots used     step    worst possible error
   8   -127 to 127   0.006   0.003
   4     -7 to 7     0.117   0.059
   2     -1 to 1     0.820   0.410

weight      8-bit     4-bit     2-bit
  0.31     0.310     0.351     0.000
 -0.82    -0.820    -0.820    -0.820
  0.07     0.071     0.117     0.000
 -0.45    -0.452    -0.469    -0.820
  0.66     0.659     0.703     0.820
```

At 8 bits, every weight barely moves. At 2 bits, every weight becomes one of only
three values: -0.82, 0 or 0.82. This is called **ternary** quantization.

### Why does this hurt a network?

A neuron multiplies inputs by weights and adds them up. Take two weights, 0.31 and -0.82,
with inputs 2 and 1:

```
true weights:   2 × 0.31  + 1 × (-0.82)  =  -0.20
8-bit weights:  2 × 0.310 + 1 × (-0.82)  =  -0.20
4-bit weights:  2 × 0.351 + 1 × (-0.82)  =  -0.12
2-bit weights:  2 × 0     + 1 × (-0.82)  =  -0.82
```

### Then why do quantized networks work at all?

Three reasons, in plain words:

1. **Errors cancel.** Rounding to the nearest step moves some weights up and some down.
   Like a diversified portfolio, 100 random errors add up to about 10× one error, not 100×.
2. **Layers reset.** Normalization layers rescale numbers after each layer,
   so a small extra gain can't compound layer after layer.
3. **The answer only needs to win.** If "cat" scores 0.70 and "dog" 0.20, small errors
   won't change the winner. Only close calls flip.

That's why 8 bits is nearly free, while 2 bits usually needs extra tricks.

---

## Day 3: Fewer bits can delete weights

Look again at the 2-bit column above. Two weights became exactly `0.000`. Why?

### The zero zone

A weight lands on slot 0 when it is less than half a step away from 0.
Take 0.31 at 2 bits, where one step is 0.82:

```
0.31 / 0.82 = 0.38   →   less than 0.5   →   slot 0
```

So every weight inside `-half a step … +half a step` vanishes.
Taller steps make a wider zone:

```
      zero zone at 4 bits:     ├┤          (-0.059 … 0.059)
      zero zone at 2 bits:  ├────────┤     (-0.410 … 0.410)
   ──────────────────────────0──────────────
```

### The result

Run `python zero_weights.py`:

```
bits    step   zero zone           weights turned to zero
   8   0.006   -0.003 to 0.003     0 of 5  []
   4   0.117   -0.059 to 0.059     0 of 5  []
   2   0.820   -0.410 to 0.410     2 of 5  [0.31, 0.07]
```

Note that -0.45 survives. It is 0.55 steps from zero, just past the halfway mark.

### Why it matters

A zero weight does nothing, so we can skip it. That's **pruning** (Lecture 3),
and here it happens for free. The flip side: if a small weight was important,
2 bits deletes it without asking.

---

## Coming next

- Day 4: K-means quantization.
- Day 5: a zero point, so we stop wasting the bottom slot.
