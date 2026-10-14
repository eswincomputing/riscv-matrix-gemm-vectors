# Method

This document explains how the reference values and the pass criteria are defined,
why they are defined that way, and what the criteria can and cannot detect. Every
number quoted here can be reproduced from the data in this repository with
`tests/selftest.py` and `check.py`.

## 1. What is fixed and what is left to the implementation

The operation under test is `C = A · Bᵀ` with BF16 inputs and an FP32 accumulator:

```
C[m][n] = Σ_k  A[m][k] · B[n][k]          k = 0 … K-1
```

Two things are fixed:

* **Each product is exact.** A BF16 value has an 8-bit significand, so the product of
  two BF16 values has at most 16 significant bits and is exactly representable in
  FP32.
* **The accumulation is in FP32,** with round-to-nearest-even in the cases provided here.

The order of the additions is not fixed. The matrix extension specification these
cases are written for states that "the reduction is done as an unordered-sum" and
that rounding is done after addition to the accumulator. This leaves the
implementation free to choose

* the order in which the products are added,
* where rounding takes place and how many times (for example after every addition,
  or once per group of products that are first summed exactly),
* the precision of intermediate sums, as long as it is at least that of FP32.

Different choices give different low-order bits. Two compliant designs therefore do
not produce the same FP32 result in general, and comparing a design bit for bit
with another design (a simulator, a C model, a previous generation) reports
failures that are not errors.

Case 008 shows the size of the effect. An implementation that rounds once per group
of 16 products and one that rounds once per group of 32 agree bit for bit on 2,950
of the 16,384 outputs (18%); 4,405 differ by 1 ULP and 9,029 by more, up to
23,774 ULP. Both are correct.

## 2. Why a tolerance in ULPs does not work

A natural fix is to accept a few ULPs of difference. The ULP is measured at the final
value, but rounding errors are made at the magnitude of the partial sums. When
positive and negative products cancel, the final value can be orders of magnitude
smaller than the partial sums.

Output (83, 55) of case 008 has an exact sum of 5.4206. Its running partial sum
reaches 41,819 on the way, and the sum of the product magnitudes is 177,205. An
implementation that rounds once per 16 products returns a value 0.0059 away from the
exact sum. Near 5.42 one ULP is 4.8 × 10⁻⁷, so the difference is 12,348 ULP. Measured
against the magnitudes that were actually rounded, it is 0.56 × 2⁻²⁴ · Σ|a·b|, an
ordinary rounding error. A ULP tolerance has to be either too tight for such outputs
or loose enough to hide real faults elsewhere.

The ratio `κ = Σ|a·b| / |C*|` (the *cancellation ratio*) measures this. Each case
records its median and maximum in `meta.json`.

## 3. Reference

The reference contains only the fixed part of the arithmetic:

```
C*[m][n]    = Σ_k A[m][k] · B[n][k]            exact sum of the products
absum[m][n] = Σ_k |A[m][k] · B[n][k]|          magnitude of what is being summed
```

Both are computed with a correctly rounded summation (`math.fsum`) and stored as
binary64 (`c_exact.bin`, `c_absum.bin`). The only error is the final conversion to
binary64, at most 2⁻⁵³ · absum, far below any tolerance used here. The reference is
the same for every implementation and does not change when the hardware changes.

## 4. Criterion

```
|Ĉ − C*|  ≤  γ(G) · Σ|a·b|,        γ(G) = G·u / (1 − G·u)
```

`u` is the unit roundoff of the accumulator: 2⁻²⁴ for FP32 with round-to-nearest
(2⁻ᵖ for a p-bit significand). `G` is **the largest number of roundings that any single
product goes through** between being formed and reaching the stored result.

Derivation. Every rounded addition returns `(x + y)(1 + δ)` with `|δ| ≤ u`. Following
each product through the additions it takes part in, the computed result is

```
Ĉ = Σ_k  p_k · Π_{j ∈ R_k} (1 + δ_j),        |R_k| ≤ G
```

where `R_k` is the set of roundings that product `k` passes through. Since
`|Π (1 + δ_j) − 1| ≤ γ(|R_k|) ≤ γ(G)` (Higham, Lemma 3.1),

```
|Ĉ − C*| = |Σ_k p_k (Π(1 + δ_j) − 1)|  ≤  γ(G) · Σ_k |p_k|.
```

The order of the additions does not appear in the bound; only `G` and `u` do. To first
order `γ(G) ≈ G·u`. The derivation assumes no overflow and no underflow, which holds
for all data in this repository.

### Choosing G

| Accumulation scheme | G |
|---|---|
| Any order, every addition rounded to FP32 (sequential chain, any tree) | K (the default) |
| Products summed exactly in groups of S, one rounding per group | ⌈K / S⌉ |
| Balanced pairwise tree, every node rounded | ⌈log₂ K⌉ + 1 |

`G = K` holds for every implementation that accumulates in FP32, whatever it does
internally, so it is the default of `check.py` and of the on-target self-check.
An implementation that documents its rounding count can use it for a tighter check
(`check.py --group S`, `--rounds G`, `set_tolerance.py`). With the data in this
repository, the largest error of any of the simulated compliant schemes is 0.92 of the
bound at its own tight G (groups of 32 on case 006, where G = 1); at G = K every
scheme stays below 0.09 of the bound.

## 5. What the criterion detects

A deviation of relative size `ε` at an output with cancellation ratio `κ` exceeds the
bound when `ε > γ(G) · κ`. For K = 1024 and G = K, `γ(G) ≈ 6.1 × 10⁻⁵`: an error of
1% is detected wherever κ is below about 160. A wrong product, a missing instruction,
an accumulator that was not cleared, swapped operands or a wrong stride change the
result by far more than the bound. The self-test injects these faults, where the
shape of the case allows them, and checks that each one is reported.

The criterion is designed to accept every compliant rounding behaviour, so it cannot
tell which one an implementation uses. Truncation instead of round-to-nearest, or an
accumulator that is one or two bits too narrow, stays within the bound (on case 008,
truncation reaches 0.44 of the bound at G = K). Two bit-exact cases cover these:

* **009 `exact_integer`.** All products are integers and `Σ|a·b| < 2²⁴` for every
  output, so every partial sum, in any order, is an integer below 2²⁴ and exactly
  representable in FP32: no rounding ever happens. A quarter of the outputs lie in
  [2²³, 2²⁴), and 2,027 of them are odd, so they need all 24 significant bits. Any
  compliant implementation returns exactly the expected bits; a narrower accumulator
  or any data-path fault does not.
* **010 `single_rounding`.** Each output has exactly two non-zero products, placed in
  different instructions. Additions of zero are exact, so exactly one addition can be
  inexact whatever the order or grouping, and the expected result is the
  round-to-nearest-even of the exact sum. The second product is set relative to the
  first so that the sum is exact with 24 bits, a tie with an even or an odd neighbour,
  just above or just below the halfway point, far below the last bit (sticky), or an
  exact cancellation to +0. Of the 16,384 outputs, truncation changes 5,683, rounding
  ties away from zero 2,238, rounding up 5,673, rounding down 5,834 and a 23-bit
  accumulator 5,022. `check.py` names the rounding behaviour that matches a failing
  dump.

## 6. Test data

Random test data need both signs. With data that are mostly positive, the partial
sums only grow and there is little cancellation. For uniform data on [−1, 3) with
K = 1024, κ stays below 1.9, two compliant implementations differ by at most 8 ULP,
and a sequential chain is at most 20 ULP from the correctly rounded sum. A ULP
tolerance then appears to work, and the problem described in section 2 does not
show up.

Cases 001–007 use zero-mean data whose magnitude also depends on the element
position, so that a transposed or shifted access changes the result. Their maximum
cancellation ratios range from about 1,900 to 10⁶. Case 008 adds rare large outliers
(±6912 in about 1% of the entries), as seen in some channels of LLM activations; its
other entries are uniform on [−1, 3). Its data are kept fixed so that results
reported for this case stay reproducible.

All data come from a fixed-seed xorshift generator and are reproduced bit for bit by
`generator/gen_vectors.py` on any machine.

## 7. Related work

The bound is the standard error bound for floating-point summation and matrix
multiplication. Normalising GEMM errors by `Σ|a|·|b|` is also how the reference BLAS
test programs check matrix products. What this repository adds is the use of that
bound as the conformance criterion for a matrix extension, with the rounding count
`G` supplied by the implementation, together with test data that exercise
cancellation and bit-exact cases for the properties the bound cannot see.

* N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002 —
  Chapter 3 (rounding error lemmas) and Chapter 4 (summation).
* P. Blanchard, N. J. Higham, F. Lopez, T. Mary, S. Pranesh, "Mixed precision block
  fused multiply-add: error analysis and application to GPU tensor cores",
  *SIAM J. Sci. Comput.* 42(3), C124–C141, 2020.
* Reference BLAS level-3 test program (`dblat3.f`, routine `DMMCH`), which compares
  computed matrix products using errors scaled by `Σ|a|·|b|`.
* RISC-V Matrix Extension Specification Proposal, version 0.6.0,
  <https://riscv.atlassian.net/wiki/spaces/AMEX/pages/893681697/Proposals> — the instructions and
  CSRs used by the test programs, and the unordered-sum semantics quoted in section 1.

---

Copyright 2026 Beijing ESWIN Computing Technology Co., Ltd.
Licensed under the Apache License, Version 2.0; see [LICENSE](../LICENSE).
