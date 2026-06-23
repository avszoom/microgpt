# MicroGPT — Concepts (the things we discussed)

This file explains, in simple words, how MicroGPT works inside. It is written so you
can come back later and quickly remember the ideas.

---

## 1. The big picture

A GPT model has one job: **look at some text and guess the next token.**
That is the whole thing. Writing text = guess the next token, add it, guess again.

The flow for one token:

```
token id (a number)  →  vector of 384 numbers  →  6 transformer blocks  →  8192 scores
   (which word)            (its meaning)            (think about it)        (next-word guess)
```

The vector is **384 numbers wide from start to end**. The blocks do not make it bigger;
they make its contents smarter. Only the last step turns 384 into 8192 scores.

---

## 2. The tokenizer (turning text into numbers)

Computers only do math on numbers, so we turn text into a list of integers.

We use **BPE (Byte-Pair Encoding)**:
- Start with single bytes (256 base tokens). This means no word is ever "unknown".
- Look at the data and **glue the most common pairs together** into bigger tokens.
- Repeat until we have our target size (we used **8192** tokens).

Result: common words become one token; rare words are split into pieces. Short and safe.

---

## 3. Embeddings (giving each token meaning)

Each token id is looked up in a table and becomes a **384-number vector**.

- At the start of training these numbers are **random** (no meaning).
- Training nudges them. Words that behave alike slowly get similar vectors.
- **Nobody programs the meaning. It is learned from data.**

We also add a **position vector** so the model knows *where* each token sits:

```
final vector = token_embedding (what the word is) + position_embedding (where it is)
```

---

## 4. Self-attention: how Key / Query / Value work

This is the key idea of transformers. For each word, the model asks:
*"Which earlier words should I look at to guess what comes next?"*

Each word makes three vectors (using learned matrices):

| Name  | Question it answers      | Simple analogy        |
|-------|--------------------------|-----------------------|
| Query | "What am I looking for?" | your search box       |
| Key   | "What do I offer?"       | a page title          |
| Value | "My actual content"      | the page's text       |

Steps for the word "sat" in "the cat sat":
1. "sat" makes a **Query**. Every earlier word makes a **Key** and a **Value**.
2. Score = Query · Key (a dot product). Bigger = better match.
3. Turn scores into percentages (softmax). E.g. "cat" gets 85%.
4. Mix the **Values** by those percentages → the new vector for "sat".

So "sat" pulls in information from the words it cares about. That is "attention".

**Causal rule:** a word can only look at words *before* it, never ahead. This stops the
model from cheating during training (the real next word is right there).

---

## 5. Multi-head attention

One attention learns *one* kind of relationship. Language has many at once (grammar,
references, position...). So we run attention several times in parallel = **heads**.

Important: the 384 numbers are **split** across heads, not copied.

```
384 = 6 heads × 64 numbers each
```

- Each head does the full Query/Key/Value process inside its own 64-number slice.
- Each head outputs **64** numbers.
- We **join** the 6 outputs back into one **384** vector (not six 384 vectors).
- A final matrix mixes the heads together.

Each head specializes during training (one tracks grammar, another tracks the previous
word, etc.). More heads add no extra parameters; they reorganize the same matrix.

---

## 6. The QKV matrix sizes

Each of the Query, Key, and Value weight matrices is **384 × 384**.

```
input vector (384)  ×  W_q (384 × 384)  =  query vector (384)
```

- The matrix is full size 384×384. The "6 heads" come from *reshaping* the 384 output
  into 6 slices of 64 — not from shrinking the matrix.
- There is also an **output projection** matrix (384 × 384) that mixes the heads after
  attention. So attention has 4 matrices: Q, K, V, and the output mix.

---

## 7. The transformer block (repeated 6 times)

Each block does two things, and the vector stays 384 wide the whole time:

```
x = x + attention(x)   # 1. COMMUNICATE: gather info from other words
x = x + mlp(x)         # 2. COMPUTE: think about it, each word on its own
```

- **Attention** = words talk to each other (the new idea in transformers).
- **MLP** = a small ordinary neural network that processes each word's vector. It briefly
  grows 384 → 1536 → 384 to have room to compute.
- **LayerNorm** keeps the numbers in a stable range so training does not blow up.
- **Residual (the `x = x + ...`)** lets information flow through all 6 layers without
  getting lost.

We stack 6 blocks so the model can build understanding in layers.

---

## 8. Three numbers people mix up

| Number | Name              | Meaning                                   |
|--------|-------------------|-------------------------------------------|
| 8192   | vocab size        | how many different tokens exist            |
| 384    | embedding dim     | how many numbers represent each token      |
| 256    | context window    | how many tokens the model can read at once |

They are independent. Bigger vocab = richer dictionary. Bigger embedding = more detail per
token. Bigger context = longer memory.

---

## 9. Context window

The **context window** is the most tokens the model can see at once. Ours is **256**.

- A word's attention can look back at *all* earlier words, but only inside this window.
- If text is longer than 256, we drop the oldest tokens and keep the last 256.
- The hard limit exists because the **position table has only 256 rows** — there is no
  "position 257" vector, so the model literally cannot place a token beyond 256.
- Bigger windows cost a lot more compute: attention grows with length **squared** (double
  the length ≈ 4× the work).

---

## 10. KV cache (a speed trick for generation)

When generating, we add one token at a time. The naive way recomputes the Key and Value
of every past token every step — wasteful, because past tokens do not change.

The **KV cache** stores the **Key and Value vectors of all past tokens** (per layer, per
head) so we only compute them once and reuse them.

- It stores K and V. It does **not** store Queries (only the current token needs a query).
- Our MicroGPT does **not** use a KV cache yet — it recomputes each step (simpler, slower).
  Adding one is a serving optimization (this is what vLLM is famous for).

---

## 11. Where the ~14M parameters live

```
6 blocks × (Q + K + V + output-mix + MLP + 2 LayerNorms)  ≈ 10.6M  (77%)
token embedding (8192 × 384)                              ≈  3.1M
position embedding (256 × 384)                            ≈  0.1M
final LayerNorm                                           ≈  tiny
output head (lm_head)                                     =  0   (shares the embedding — "weight tying")
─────────────────────────────────────────────────────────────────
total                                                    ≈ 13.8M  (reported 13.77M)
```

Most of the "intelligence" lives in the stacked blocks, not the lookups.

---

## 12. Why this model is "dumb", and the three training stages

Our model only did **Stage 1**. It is a *base model*: it continues text like Wikipedia,
but it does not know it should *answer* you. (Ask "9+9" and it writes a ship article.)

To get ChatGPT-like behavior you add two more stages:

| Stage | Name                | What it improves                          |
|-------|---------------------|-------------------------------------------|
| 1     | Pretraining (done)  | **Knowledge** — language and patterns      |
| 2     | Instruction tuning (SFT) | **Behavior** — it learns to respond to requests |
| 3     | Preference tuning (RLHF/DPO/RLVR) | **Quality** — it gives *good*, correct answers |

Key facts we discussed:
- Stage 1 sets the **capability ceiling**. Stages 2–3 cannot add knowledge that is not
  there; they only **surface and sharpen** what pretraining already learned.
- A tiny 14M model cannot reason no matter what — reasoning needs **scale**.
- For code/math, the strongest Stage-3 method is **RLVR**: reward the model for answers
  that **pass tests** (verifiable), not just answers humans like.

The next projects (fine-tuning, preference tuning, the Qwen coding project) are exactly
Stages 2 and 3 built on top of a base model.

---

## One-line summary

> A token becomes a vector that means something → attention lets vectors share info →
> the vector becomes "this word in context" → the last step turns that into a guess for
> the next word. Train it on lots of text and it learns to write.
