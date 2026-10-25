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
python test_quantize.py    # run the 10 checks
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

## Coming next

- Day 2: fewer bits (4-bit, 2-bit). Watch the error grow.
- Day 3: K-means quantization.
- Day 4: binary and ternary weights (-1, 0, +1).
