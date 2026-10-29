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
python kmeans_quantize.py  # let the slots move to the weights
python shift_quantize.py   # start the slots at the smallest value
python test_quantize.py    # run all the checks
python test_kmeans_quantize.py
python test_shift_quantize.py
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

## Day 4: K-means quantization

Linear slots sit at equal gaps, no matter where the weights are.
At 2 bits, that wastes slots and deletes small weights:

```
weights:   -0.82      -0.45        0.07   0.31        0.66
              ●          ●           ●     ●            ●
  ────────────┼──────────────────────┼──────────────────┼────
linear:     -0.82                    0                 0.82
```

**K-means lets the slots move to where the weights bunch up.**

### Ice cream carts on a beach

Weights are kids. Slots are ice cream carts. Repeat two rules:

```
Rule 1: each kid walks to the NEAREST cart.
Rule 2: each cart moves to the AVERAGE of its kids.
Stop when no kid switches carts.
```

Kids at 0, 4, 10, 16. Carts start at 0 and 3:

```
Round 1:  groups {0} {4, 10, 16}   →  carts move to 0 and 10
Round 2:  kid 4 switches to A      →  carts move to 2 and 13
Round 3:  nobody switches          →  stop
```

We repeat because moving a cart changes who is nearest.

### The result

Start from the linear slots and run `kmeans`:

```
weight    linear slot   error    k-means slot   error
-0.82       -0.82       0.00       -0.635       0.185
-0.45       -0.82       0.37       -0.635       0.185
 0.07        0          0.07        0.19        0.12
 0.31        0          0.31        0.19        0.12
 0.66        0.82       0.16        0.66        0.00
─────────────────────────────────────────────────────
worst error             0.37                    0.185
```

The worst error halves, and no weight vanishes.

Each weight now stores only a 2-bit label (0, 1 or 2). The real values live
once, in a short list called the **codebook**: `[-0.635, 0.19, 0.66]`.
For a million weights, that list costs almost nothing.

### Two things to know

- **It always stops.** Each rule can only shrink the total distance.
- **The start matters.** Carts starting at 5 and 16 get stuck at 4.67 and 16,
  a worse answer. Starting from the linear slots spreads the carts out evenly.

Try it: `python kmeans_quantize.py`, then `python test_kmeans_quantize.py`.

### A question to think about

K-means only ever takes small steps that make things a little better.
Like a hiker walking downhill in fog, it stops in the first valley it finds,
even if a deeper valley sits just over the next hill.

```
start [0, 3]    →  [2, 13]       total walk 10     (deeper valley)
start [5, 16]   →  [4.67, 16]    total walk 10.7   (stuck in a shallow one)
```

**You can't know the best start ahead of time. So how could you make
getting stuck less likely?** (Answer at the start of Day 5.)

---

## Day 5: Shift the staircase

**Answer to Day 4's question first.** Try several starts, and keep the one
with the smallest total walk. This is called **restarts**. Each try is fast
and independent, so they can even run at the same time.

### The problem: empty slots

So far, our slots sat around 0. That fits weights like `-0.82 … 0.66`.
But some numbers in a network never go near 0. Say the values are `8 … 14`.

With 3 bits we get 8 slots, so 7 steps. Starting the staircase at 0 means
climbing from 0 to 14:

```
START AT 0                       START AT THE SMALLEST VALUE (8)

14 ┤ ■                           14 ┤ ■
12 ┤ ■                              ┤ ■
10 ┤ ■                              ┤ ■
 8 ┤ ■                              ┤ ■
 6 ┤ □  ← no value                  ┤ ■
 4 ┤ □    lives here                ┤ ■
 2 ┤ □                              ┤ ■
 0 ┤ □                            8 ┤ ■

step = 14 ÷ 7 = 2                step = 6 ÷ 7 = 0.857
```

Half the slots are wasted on the left. **The fix: start the staircase at the
smallest value**, so every step lands where the values actually are.

### Two notes are all you need

For values `8 … 22` and 3 bits:

```
start = smallest value             = 8
step  = (biggest − smallest) ÷ 7   = 14 ÷ 7 = 2
```

(3 bits give 2 × 2 × 2 = 8 slots. 8 slots have 7 steps between them.)

```
slot:    0    1    2    3    4    5    6    7
value:   8   10   12   14   16   18   20   22
```

Going there and back:

```
value → slot:   round((value − start) ÷ step)     14 → (14 − 8) ÷ 2 = 3
slot → value:   start + step × slot               3  → 8 + 2 × 3    = 14
```

We store only the tiny slots, plus the two notes once. When the model runs,
slots turn back into values just before the math.

### The result

Values `8 … 14`, 3 bits, worst error over 601 values:

```
Day 1 method     ████████████████████████  2.33
Shift (today)    ████▌                     0.43
```

About 5× smaller, with the same 3 bits. The worst error is still half a step.

**One trap:** if every value is the same (`5, 5, 5`), the step is 0, and
dividing by it crashes. Every value sits on slot 0 then, so we use step 1.

Try it: `python shift_quantize.py`, then `python test_shift_quantize.py`.

### A note on the name

Textbooks call this *asymmetric quantization*. Instead of the start, they store
the **zero point**: the slot where real 0 lands. For values `-2 … 12` with
step 2, real 0 sits on slot 1. It's the same information, written another way.

> One can be smart about memory and computation with little tricks that
> seem obvious in hindsight but add a lot.

### A question to think about

Networks are full of exact zeros. Every negative value that goes through a
ReLU comes out as exactly 0. Now take values `-1 … 13` with 3 bits:

```
step = (13 − (−1)) ÷ 7 = 2

slot:    0    1    2    3    4    5    6    7
value:  -1    1    3    5    7    9   11   13
            ↑
        real 0 falls between two slots
```

No slot holds 0. So every 0 comes back as -1 or 1.

**Is that a problem, if the error is still only half a step? And how could
you move the staircase a tiny bit so that 0 lands exactly on a slot?**
(Day 6 starts here.)

---

## Coming next

- Why textbooks store the zero point instead of the start.
- Quantize a real tiny model and compare its outputs.
